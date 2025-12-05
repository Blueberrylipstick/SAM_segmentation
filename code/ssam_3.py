import ast
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import torch
from transformers import Sam3Model, Sam3Processor


# =============================================================================
# CONFIG
# =============================================================================

# Path to your annotations CSV
CSV_PATH = "/Users/yulch/Desktop/sam/annotations_u45_Flinders_Western_Boundary_2011_targeted_Flinders.csv"

# Output directory
OUTPUT_DIR = "results_sam3_exemplarBest"

# Exemplar bank config:
#   MODE:
#     0 = no exemplar bank (disable exemplar-based prompts)
#     1 = small hardcoded taxonomy dict (Sponges, Cnidaria, etc.)
#     3 = load from exemplar_bank.pkl (your file built from CSV)
EXEMPLAR_BANK_MODE = 3
EXEMPLAR_BANK_PATH = "/Users/yulch/Desktop/RAI/leaf_exemplar_bank_best.pkl"  # <- change if needed


# =============================================================================
# UTILS: POLYGON, IOU
# =============================================================================

def polygon_to_mask(relative_polygon, center_norm, height, width):
    """
    Convert a polygon given as *relative normalized offsets* around the point center
    into a binary mask.

    relative_polygon: list [[dx, dy], ...] with values roughly in [-0.2, 0.2]
    center_norm: [cx, cy] in [0,1] (this is point_x, point_y)
    """
    try:
        if isinstance(relative_polygon, str):
            relative_polygon = ast.literal_eval(relative_polygon)

        cx, cy = center_norm
        poly_px = []

        for dx, dy in relative_polygon:
            x_norm = cx + dx
            y_norm = cy + dy

            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            px = int(x_norm * width)
            py = int(y_norm * height)
            poly_px.append([px, py])

        poly_px = np.array(poly_px, dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_px], 1)

        return mask.astype(bool)

    except Exception as e:
        print(f"Error converting polygon to mask: {e}")
        return None


def compute_iou(mask_pred, mask_gt):
    """
    Compute IoU between predicted and GT masks.
    """
    if mask_gt is None:
        return 0.0

    pred = (mask_pred > 0.5)
    gt = mask_gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


# =============================================================================
# EXEMPLAR BANK LOADING (MODE 1 / MODE 3)
# =============================================================================

def load_hardcoded_exemplar_bank():
    """
    MODE 1: Tiny manual exemplar bank with taxonomy-level concepts.
    Keys are mid-level concepts; values contain prompt_text.
    """
    bank = {
        "Sponges": {
            "prompt_text": "sponges",
            "description": "All sponge-like sessile organisms",
        },
        "Cnidaria": {
            "prompt_text": "cnidarian corals and anemones",
            "description": "Corals, anemones and other cnidarians",
        },
        "Molluscs": {
            "prompt_text": "molluscs",
            "description": "Shell-bearing molluscs and similar animals",
        },
        "Echinoderms": {
            "prompt_text": "echinoderms",
            "description": "Starfish, sea urchins and other echinoderms",
        },
        "Substrate": {
            "prompt_text": "seafloor substrate",
            "description": "Bare seafloor sediments",
        },
        "Relief": {
            "prompt_text": "rocky reef relief",
            "description": "Rocky outcrops and hard relief structures",
        },
    }
    return bank


def load_pkl_exemplar_bank(path):
    """
    MODE 3: Load your exemplar_bank.pkl built from CSV.

    The original builder stored:
        exemplar_bank[leaf_label] = {
            "image": image,                # np.array
            "mask": mask.astype(np.uint8), # 0/1 mask
            "image_path": image_path,
            "full_lineage": full_lineage,
            "label_name": simple_label,
            "center_norm": center_norm,
        }

    For OPTION A, we **only need text**. So we convert to a simple dict:
        {
          key (leaf_label): {
             "prompt_text": label_name or leaf_label,
             "description": full_lineage or label_name
          }
        }
    """
    path = Path(path)
    if not path.exists():
        print(f"⚠ Exemplar bank .pkl not found at {path}, falling back to empty bank.")
        return {}

    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        print(f"⚠ Failed to load exemplar bank pkl: {e}")
        return {}

    simple_bank = {}
    for key, entry in raw.items():
        label_name = str(entry.get("label_name", key))
        full_lineage = entry.get("full_lineage", None)
        desc = str(full_lineage) if full_lineage is not None else label_name
        simple_bank[key] = {
            "prompt_text": label_name,
            "description": desc,
        }

    print(f"✓ Loaded exemplar bank from {path} with {len(simple_bank)} leaf concepts")
    return simple_bank


def load_exemplar_bank(mode, pkl_path=None):
    """
    Wrapper to choose between:
      - 0: no exemplar bank
      - 1: hardcoded taxonomy bank
      - 3: load from exemplar_bank.pkl
    """
    if mode == 0:
        print("Exemplar bank mode 0: disabled")
        return {}
    if mode == 1:
        print("Exemplar bank mode 1: hardcoded taxonomy concepts")
        return load_hardcoded_exemplar_bank()
    if mode == 3:
        print(f"Exemplar bank mode 3: loading from {pkl_path}")
        return load_pkl_exemplar_bank(pkl_path or EXEMPLAR_BANK_PATH)

    print(f"⚠ Unknown EXEMPLAR_BANK_MODE={mode}, disabling exemplar bank.")
    return {}


# =============================================================================
# EXEMPLAR-BASED TEXT PROMPT SELECTION (OPTION A)
# =============================================================================

def clean_taxon_token(token: str) -> str:
    """
    Light cleaning: remove leading numbering like '1.1', punctuation, lowercase.
    Example:
      '1.1 Biota' -> 'biota'
    """
    token = token.strip()
    parts = token.split(" ", 1)
    if len(parts) == 2 and any(ch.isdigit() for ch in parts[0]):
        token = parts[1]
    token = token.replace(".", " ").replace(",", " ")
    token = " ".join(token.split())
    return token.lower()


def pick_exemplar_prompt_from_lineage(lineage, exemplar_bank):
    """
    Exemplar-based text prompt selection (A2 style):

      - Take taxonomy string like "1.1 Biota > Sponges > Cup-likes > Cups > Fan Pink"
      - Split by '>'
      - Walk from deepest to shallower levels.
      - For each taxon token, check if it contains any exemplar key (case-insensitive).
        * Example: key "sponges" will match taxon token "Sponges".

    Returns:
      (prompt_text, exemplar_key) or (None, None) if nothing matched.
    """
    if not exemplar_bank:
        return None, None

    if lineage is None:
        return None, None

    lineage_str = str(lineage)
    if not lineage_str or lineage_str.lower() == "nan":
        return None, None

    parts = [p.strip() for p in lineage_str.split(">") if p.strip()]
    if not parts:
        return None, None

    # Cache cleaned keys
    ex_clean_map = {
        key: key.lower()
        for key in exemplar_bank.keys()
    }

    for part in reversed(parts):  # deepest to top
        part_clean = clean_taxon_token(part)
        for key, key_clean in ex_clean_map.items():
            if key_clean in part_clean:
                ex_info = exemplar_bank[key]
                prompt_text = ex_info.get("prompt_text", key)
                return prompt_text, key

    return None, None


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class UnderwaterSAM3Pipeline:
    """
    SAM3 pipeline with OPTION A:

      PRIORITY:
        1) Exemplar-based concept prompt from taxonomy (if exemplar_bank provided).
        2) Fallback: heuristic text prompt from lineage/label.
        3) Final fallback: synthetic box around point.

      For each object:
        * Build candidate text prompts:
            - (maybe) exemplar prompt from taxonomy
            - default lineage/label prompt
        * Run SAM3 with text prompts (one after another), keep BEST scoring mask.
        * If no text prompt yields a mask that covers the point, try a box-only prompt.
        * Choose the best result among text / box.
    """

    def __init__(self, model_id="facebook/sam3", device="cpu", exemplar_bank=None):
        self.device = torch.device(device)
        print(f"Loading SAM 3 model '{model_id}' on {self.device}...")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model.eval()

        self.exemplar_bank = exemplar_bank or {}
        print(f"✓ SAM 3 initialized on {self.device}")
        if self.exemplar_bank:
            print(f"✓ Text exemplar bank loaded with {len(self.exemplar_bank)} concepts")

    # ---------------------------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------------------------
    def load_image(self, image_path, max_retries=3, timeout=30):
        """
        Load image from URL or local path with robust error handling.
        """
        if isinstance(image_path, str) and image_path.startswith("http"):
            for attempt in range(max_retries):
                try:
                    print(f"    Downloading from URL (attempt {attempt + 1}/{max_retries})...")
                    response = requests.get(
                        image_path,
                        timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    response.raise_for_status()

                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode image data")

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"    ✓ Successfully loaded image from URL")
                    return image

                except Exception as e:
                    print(f"    ✗ Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to download image after {max_retries} attempts: {e}")
            return None
        else:
            try:
                if not Path(image_path).exists():
                    raise ValueError(f"Local file does not exist: {image_path}")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to read image file: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"    ✓ Successfully loaded local image")
                return image
            except Exception as e:
                raise ValueError(f"Failed to load local image: {e}")

    # ---------------------------------------------------------------------
    # TEXT PROMPT BUILDERS
    # ---------------------------------------------------------------------
    @staticmethod
    def build_text_prompt(label_name, lineage):
        """
        Heuristic text prompt builder:
          - Try mid-level taxonomy from lineage (2nd/3rd part).
          - Fallback to label_name.
        """
        if lineage is not None and isinstance(lineage, str) and len(lineage.strip()) > 0:
            try:
                parts = [p.strip() for p in lineage.split(">")]
                # Example: "1.1 Biota > Sponges > Cup-likes > Cups > Fan Pink"
                if len(parts) >= 4:
                    prompt = parts[2]  # "Cup-likes"
                elif len(parts) >= 3:
                    prompt = parts[2]
                elif len(parts) >= 2:
                    prompt = parts[1]
                else:
                    prompt = parts[-1]
                if len(prompt) >= 2:
                    return prompt
            except Exception:
                pass

        return str(label_name)

    @staticmethod
    def build_box_around_point(px, py, width, height, box_size_frac=0.18, min_box=64):
        """
        Build a square box around a point (px, py).

        box_size = max(min_box, box_size_frac * min(width, height))

        Then we take:
            x1 = px - box_size/2,  x2 = px + box_size/2
            y1 = py - box_size/2,  y2 = py + box_size/2

        and clamp the coordinates to the image bounds.
        """
        box_size = int(max(min_box, box_size_frac * min(width, height)))
        half = box_size // 2

        x1 = max(0, px - half)
        y1 = max(0, py - half)
        x2 = min(width - 1, px + half)
        y2 = min(height - 1, py + half)

        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)

        return [float(x1), float(y1), float(x2), float(y2)]

    # ---------------------------------------------------------------------
    # CORE SEGMENTATION PER OBJECT
    # ---------------------------------------------------------------------
    def segment_image_per_object_with_labels(
        self,
        image_path,
        normalized_points,
        label_names=None,
        label_lineages=None,
        is_background=None,
        polygons=None,
        multimask_output=False,
        score_threshold=0.5,
        mask_threshold=0.5,
    ):
        """
        FINAL VERSION — uses combined:
            • text prompt
            • point prompt
            • box prompt
            • num_refinement_steps
        for every object.

        Fallback order:
            1) exemplar-based text + point + box
            2) default text + point + box
            3) box-only prompt + point
        """
        # -------------------------------------------------------------
        # Load image
        # -------------------------------------------------------------
        try:
            image = self.load_image(image_path)
            height, width = image.shape[:2]
            print(f"    Image dimensions: {width}x{height}")
        except Exception as e:
            raise ValueError(f"Image loading failed: {e}")

        # Normalize inputs
        if is_background is None:
            is_background = [1] * len(normalized_points)
        if label_names is None:
            label_names = [f"Object_{i}" for i in range(len(normalized_points))]
        if label_lineages is None:
            label_lineages = [None] * len(normalized_points)
        if polygons is None:
            polygons = [None] * len(normalized_points)

        results = []
        failed_objects = []

        # =============================================================
        # MAIN LOOP — EACH OBJECT
        # =============================================================
        for idx, (norm_point, label, lineage, use_flag, poly) in enumerate(
            zip(normalized_points, label_names, label_lineages, is_background, polygons)
        ):
            if use_flag != 1:
                continue

            if not (0 <= norm_point[0] <= 1 and 0 <= norm_point[1] <= 1):
                continue

            pixel_x = int(norm_point[0] * width)
            pixel_y = int(norm_point[1] * height)

            pixel_x = max(0, min(width - 1, pixel_x))
            pixel_y = max(0, min(height - 1, pixel_y))

            print(f"\n    → Object {idx}: label='{label}', point=({norm_point[0]:.3f}, {norm_point[1]:.3f})")

            # ---------------------------------------------------------
            # 0) Build bounding box for multimodal prompting
            # ---------------------------------------------------------
            box_xyxy = self.build_box_around_point(
                pixel_x, pixel_y, width, height,
                box_size_frac=0.18,    # IMPORTANT: better IoU
                min_box=48             # IMPORTANT: more stable for underwater images
            )

            # ---------------------------------------------------------
            # 1) Create text prompt candidates
            # ---------------------------------------------------------
            text_candidates = []

            # Exemplar-first
            ex_prompt, ex_key = pick_exemplar_prompt_from_lineage(lineage, self.exemplar_bank)
            if ex_prompt is not None:
                print(f"      Exemplar concept: '{ex_key}' → prompt '{ex_prompt}'")
                text_candidates.append((ex_prompt, "exemplar"))

            # Default taxonomy-based fallback
            default_prompt = self.build_text_prompt(label, lineage)
            if default_prompt is not None:
                if not text_candidates or default_prompt.lower() != text_candidates[0][0].lower():
                    print(f"      Default text prompt: '{default_prompt}'")
                    text_candidates.append((default_prompt, "default"))

            best_text_result = None

            # =========================================================
            # 2) RUN TEXT + POINT + BOX multimodal SAM3 prompting
            # =========================================================
            for text_prompt, source in text_candidates:
                try:
                    inputs_text = self.processor(
                        images=image,
                        text=text_prompt,
                        input_points=[[[pixel_x, pixel_y]]],      # shape: [B, N, 2]
                        input_points_labels=[[[1]]],              # same shape
                        input_boxes=[[box_xyxy]],                 # shape: [B, M, 4]
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        outputs_text = self.model(
                            **inputs_text,
                            mode="prompt",
                            num_refinement_steps=3
                        )

                    post_text = self.processor.post_process_instance_segmentation(
                        outputs_text,
                        threshold=0.0,
                        mask_threshold=0.3,
                        target_sizes=inputs_text.get("original_sizes").tolist(),
                    )[0]

                    masks_t = post_text["masks"].cpu().numpy()
                    scores_t = post_text["scores"].cpu().numpy()

                    if len(masks_t) == 0:
                        print(f"      ⚠ No masks for text '{text_prompt}' ({source})")
                        continue

                    # Prefer mask covering the point
                    chosen_t = None
                    for j, m in enumerate(masks_t):
                        try:
                            if m[pixel_y, pixel_x] > 0:
                                chosen_t = j
                                break
                        except Exception:
                            continue

                    if chosen_t is None:
                        print(f"      ⚠ No mask covers the point for text prompt '{text_prompt}'")
                        continue

                    cand = {
                        "mask": masks_t[chosen_t],
                        "score": float(scores_t[chosen_t]),
                        "prompt_type": "text",
                        "text_used": text_prompt,
                        "text_source": source,
                    }
                    print(f"      ✓ Text+point+box score={cand['score']:.3f}")

                    if (best_text_result is None) or (cand["score"] > best_text_result["score"]):
                        best_text_result = cand

                except Exception as e:
                    print(f"      ✗ Text multimodal prompt failed: {e}")
                    continue

            # =========================================================
            # 3) BOX-ONLY FALLBACK (still includes a point!)
            # =========================================================
            box_result = None
            if best_text_result is None:
                print("      → Falling back: box + point prompt")
                try:
                    inputs_box = self.processor(
                        images=image,
                        input_points=[[[pixel_x, pixel_y]]],
                        input_points_labels=[[[1]]],
                        input_boxes=[[box_xyxy]],
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        outputs_box = self.model(
                            **inputs_box,
                            mode="prompt",
                            num_refinement_steps=3
                        )

                    post_box = self.processor.post_process_instance_segmentation(
                        outputs_box,
                        threshold=0.0,
                        mask_threshold=0.3,
                        target_sizes=inputs_box.get("original_sizes").tolist(),
                    )[0]

                    masks_b = post_box["masks"].cpu().numpy()
                    scores_b = post_box["scores"].cpu().numpy()

                    if len(masks_b) > 0:
                        chosen_b = None
                        for j, m in enumerate(masks_b):
                            try:
                                if m[pixel_y, pixel_x] > 0:
                                    chosen_b = j
                                    break
                            except Exception:
                                continue

                        if chosen_b is None:
                            chosen_b = int(np.argmax(scores_b))

                        box_result = {
                            "mask": masks_b[chosen_b],
                            "score": float(scores_b[chosen_b]),
                            "prompt_type": "box",
                            "box_xyxy": box_xyxy,
                        }
                        print(f"      ✓ Box+point score={box_result['score']:.3f}")

                except Exception as e:
                    print(f"      ✗ Box+point prompt failed: {e}")

            # =========================================================
            # 4) CHOOSE BEST RESULT
            # =========================================================
            if best_text_result is None and box_result is None:
                print(f"      ✗ No valid mask for object {idx}")
                failed_objects.append(idx)
                continue

            best_result = best_text_result if (
                best_text_result and (
                    box_result is None or best_text_result["score"] >= box_result["score"]
                )
            ) else box_result

            # =========================================================
            # 5) IoU EVALUATION
            # =========================================================
            gt_mask = polygon_to_mask(poly, norm_point, height, width) if poly is not None else None
            iou_value = compute_iou(best_result["mask"], gt_mask) if gt_mask is not None else 0.0

            best_result.update({
                "iou": float(iou_value),
                "normalized_point": norm_point,
                "pixel_point": [pixel_x, pixel_y],
                "label_name": label,
                "object_id": idx,
                "is_background": False,
                "success": True
            })

            results.append(best_result)

        # -------------------------------------------------------------
        # Finalize
        # -------------------------------------------------------------
        if failed_objects:
            print(f"    ⚠ Failed: {failed_objects}")

        if len(results) == 0:
            print("    ⚠ No objects segmented.")

        return results, image


    # ---------------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------------
    def visualize_individual_results(self, image, results, save_dir=None, image_name="image", show=False):
        saved_paths = []
        failed_visualizations = []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for result in results:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                axes[0].imshow(image)
                axes[0].scatter(
                    result["pixel_point"][0],
                    result["pixel_point"][1],
                    c="red",
                    s=300,
                    marker="*",
                    edgecolors="white",
                    linewidths=2,
                )
                axes[0].set_title(
                    f"Object: {result['label_name']}\nPoint: {result['pixel_point']}\nPrompt: {result.get('prompt_type')}"
                )
                axes[0].axis("off")

                axes[1].imshow(image)
                axes[1].imshow(result["mask"], alpha=0.5, cmap="jet")
                axes[1].set_title(f"Segmentation\nScore: {result['score']:.3f}, IoU: {result.get('iou', 0.0):.3f}")
                axes[1].axis("off")

                plt.tight_layout()

                if save_dir:
                    safe_label = (
                        result["label_name"].replace(" ", "_").replace("/", "_").replace("\\", "_")
                    )
                    save_path = Path(save_dir) / f"{image_name}_obj{result['object_id']:03d}_{safe_label}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    saved_paths.append(save_path)

                if show:
                    plt.show()
                else:
                    plt.close()

            except Exception as e:
                print(f"    ✗ Failed to create visualization for object {result.get('object_id', 'NA')}: {e}")
                failed_visualizations.append(result.get("object_id", -1))
                if "fig" in locals():
                    plt.close(fig)
                continue

        if failed_visualizations:
            print(f"    ⚠ Failed to create {len(failed_visualizations)} visualizations")

        return saved_paths

    def visualize_all_objects(self, image, results, save_path=None, show=False):
        try:
            num_objects = len(results)
            if num_objects == 0:
                print("    ⚠ No results to visualize")
                return None

            cols = min(4, num_objects + 1)
            rows = max(1, (num_objects + cols) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            axes[0].imshow(image)
            for result in results:
                axes[0].scatter(
                    result["pixel_point"][0],
                    result["pixel_point"][1],
                    c="red",
                    s=200,
                    marker="*",
                    edgecolors="white",
                    linewidths=2,
                )
            axes[0].set_title(f"Original Image\n{num_objects} objects")
            axes[0].axis("off")

            for idx, result in enumerate(results):
                if idx + 1 < len(axes):
                    ax = axes[idx + 1]
                    ax.imshow(image)
                    ax.imshow(result["mask"], alpha=0.5, cmap="jet")
                    ax.set_title(
                        f"{result['label_name']}\nScore: {result['score']:.3f}, IoU: {result.get('iou', 0.0):.3f}",
                        fontsize=10,
                    )
                    ax.axis("off")

            for idx in range(num_objects + 1, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"    ✓ Saved overview to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig

        except Exception as e:
            print(f"    ✗ Failed to create overview visualization: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None


# =============================================================================
# DATA PREP, SAVING
# =============================================================================

def prepare_annotations_from_df(df):
    """
    Convert annotation DataFrame to grouped format with positive/negative flags and polygons.
    """
    image_annotations = defaultdict(
        lambda: {
            "points": [],
            "label_names": [],
            "label_lineages": [],
            "is_background": [],
            "polygons": [],
            "image_path": None,
        }
    )

    invalid_rows = []

    for idx, row in df.iterrows():
        try:
            if pd.isna(row["point_media_path_best"]) or pd.isna(row["point_x"]) or pd.isna(row["point_y"]):
                invalid_rows.append(idx)
                continue

            image_path = str(row["point_media_path_best"])
            point_x = float(row["point_x"])
            point_y = float(row["point_y"])
            label_name = str(row.get("label_name", "Unknown"))
            use_flag = int(row.get("background", 1))

            if not (0 <= point_x <= 1 and 0 <= point_y <= 1):
                print(f"⚠ Warning: Invalid coordinates ({point_x}, {point_y}) for row {idx}, skipping")
                invalid_rows.append(idx)
                continue

            lineage = row.get("label_lineage_names", None)
            if pd.isna(lineage):
                lineage = row.get("label_lineage_names", None)

            image_annotations[image_path]["points"].append([point_x, point_y])
            image_annotations[image_path]["label_names"].append(label_name)
            image_annotations[image_path]["label_lineages"].append(lineage)
            image_annotations[image_path]["is_background"].append(use_flag)
            image_annotations[image_path]["polygons"].append(row["point_polygon"])
            image_annotations[image_path]["image_path"] = image_path

        except Exception as e:
            print(f"✗ Error processing row {idx}: {e}")
            invalid_rows.append(idx)
            continue

    if invalid_rows:
        print(f"⚠ Skipped {len(invalid_rows)} invalid rows")

    return dict(image_annotations)


def save_individual_masks(results, output_dir, image_name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []
    failed_saves = []

    for result in results:
        try:
            safe_label = result["label_name"].replace(" ", "_").replace("/", "_").replace("\\", "_")
            mask_path = Path(output_dir) / f"{image_name}_obj{result['object_id']:03d}_{safe_label}_mask.npy"
            np.save(mask_path, result["mask"])
            saved_paths.append(mask_path)
        except Exception as e:
            print(f"    ✗ Failed to save mask for object {result['object_id']}: {e}")
            failed_saves.append(result["object_id"])
            continue

    if failed_saves:
        print(f"    ⚠ Failed to save {len(failed_saves)} masks")

    return saved_paths


def save_results_metadata(results, output_path, image_name, image_path, time_per_object_sec=None):
    """
    Save metadata about segmentation results to CSV, including IoU.
    """
    try:
        metadata = []

        for result in results:
            row = {
                "image_name": image_name,
                "image_path": image_path,
                "object_id": result["object_id"],
                "label_name": result["label_name"],
                "point_x_normalized": result["normalized_point"][0],
                "point_y_normalized": result["normalized_point"][1],
                "point_x_pixel": result["pixel_point"][0],
                "point_y_pixel": result["pixel_point"][1],
                "segmentation_score": result["score"],
                "gt_iou": result.get("iou", 0.0),
                "mask_shape": str(result["mask"].shape),
                "is_background": result.get("is_background", False),
                "prompt_type": result.get("prompt_type", None),
                "text_prompt": result.get("text_used", None),
                "text_source": result.get("text_source", None),
                "success": result.get("success", True),
            }
            if time_per_object_sec is not None:
                row["time_per_object_sec"] = time_per_object_sec

            metadata.append(row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)

        return df

    except Exception as e:
        print(f"    ✗ Failed to save metadata: {e}")
        return None


# =============================================================================
# SINGLE IMAGE + DATASET PROCESSING
# =============================================================================

def process_single_image(
    pipeline,
    image_data,
    output_dir=OUTPUT_DIR,
    visualize_individual=True,
    visualize_overview=True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_path = image_data["image_path"]
    image_name = Path(image_path).stem if not str(image_path).startswith("http") else f"image_{abs(hash(image_path)) % 100000}"

    total_points = len(image_data["points"])
    use_flags = image_data.get("is_background", [1] * total_points)
    num_positive = sum(use_flags)
    num_negative = total_points - num_positive

    print(f"Processing: {image_name}")
    print(f"  Total points: {total_points} (Positive: {num_positive}, Ignored: {num_negative})")

    start_time = time.perf_counter()

    try:
        results, image = pipeline.segment_image_per_object_with_labels(
            image_path=image_path,
            normalized_points=image_data["points"],
            label_names=image_data["label_names"],
            label_lineages=image_data.get("label_lineages"),
            is_background=image_data.get("is_background"),
            polygons=image_data.get("polygons"),
            multimask_output=False,
        )

        print(f"  ✓ Successfully segmented {len(results)} positive objects")

        masks_dir = Path(output_dir) / "masks"
        vis_dir = Path(output_dir) / "visualizations"
        metadata_dir = Path(output_dir) / "metadata"

        try:
            mask_paths = save_individual_masks(results, masks_dir, image_name)
            print(f"  ✓ Saved {len(mask_paths)} masks")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to save some masks: {e}")

        if visualize_individual:
            try:
                vis_paths = pipeline.visualize_individual_results(
                    image=image,
                    results=results,
                    save_dir=vis_dir / "individual",
                    image_name=image_name,
                    show=False,
                )
                print(f"  ✓ Created {len(vis_paths)} visualizations")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to create visualizations: {e}")

        if visualize_overview:
            try:
                overview_path = vis_dir / "overview" / f"{image_name}_overview.png"
                pipeline.visualize_all_objects(
                    image=image,
                    results=results,
                    save_path=overview_path,
                    show=False,
                )
                print(f"  ✓ Created overview")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to create overview: {e}")

        scores = [r["score"] for r in results]
        ious = [r.get("iou", 0.0) for r in results]

        end_time = time.perf_counter()
        total_time = end_time - start_time
        num_objects = len(results)
        time_per_object = total_time / num_objects if num_objects else 0.0

        print(f"  Processing time: {total_time:.3f} s ({time_per_object:.3f} s per object)")

        try:
            metadata_path = metadata_dir / f"{image_name}_metadata.csv"
            metadata_df = save_results_metadata(
                results,
                metadata_path,
                image_name,
                image_path,
                time_per_object_sec=time_per_object,
            )
            if metadata_df is not None:
                print(f"  ✓ Saved metadata")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to save metadata: {e}")

        summary = {
            "image_path": image_path,
            "image_name": image_name,
            "num_points_total": total_points,
            "num_positive": num_positive,
            "num_negative": num_negative,
            "num_objects_segmented": len(results),
            "mean_score": np.mean(scores) if scores else 0,
            "min_score": np.min(scores) if scores else 0,
            "max_score": np.max(scores) if scores else 0,
            "mean_iou": np.mean(ious) if ious else 0,
            "min_iou": np.min(ious) if ious else 0,
            "max_iou": np.max(ious) if ious else 0,
            "total_time_sec": total_time,
            "time_per_object_sec": time_per_object,
            "success": True,
            "error": None,
        }

        print(f"  ✓ Success! {len(results)} positive objects segmented")
        if scores:
            print(
                f"    Mean score: {summary['mean_score']:.3f}, "
                f"Range: [{summary['min_score']:.3f}, {summary['max_score']:.3f}]"
            )
        if ious:
            print(
                f"    Mean IoU:   {summary['mean_iou']:.3f}, "
                f"Range: [{summary['min_iou']:.3f}, {summary['max_iou']:.3f}]"
            )

        return summary

    except Exception as e:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"  ✗ Failed to process image: {e}")
        import traceback

        print(f"  Error details: {traceback.format_exc()}")

        return {
            "image_path": image_path,
            "image_name": image_name,
            "num_points_total": total_points,
            "num_positive": num_positive,
            "num_negative": num_negative,
            "num_objects_segmented": 0,
            "mean_score": 0,
            "min_score": 0,
            "max_score": 0,
            "mean_iou": 0,
            "min_iou": 0,
            "max_iou": 0,
            "total_time_sec": total_time,
            "time_per_object_sec": 0.0,
            "success": False,
            "error": str(e),
        }


def process_dataset(
    pipeline,
    image_annotations,
    output_dir=OUTPUT_DIR,
    max_images=None,
    visualize_individual=True,
    visualize_overview=True,
    continue_on_failure=True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    image_items = list(image_annotations.items())

    if max_images:
        image_items = image_items[:max_images]
        print(f"Processing limited to {max_images} images (set max_images=None for full dataset)")

    total_points = sum(len(data["points"]) for _, data in image_items)
    total_positive = sum(sum(data.get("is_background", [])) for _, data in image_items)
    total_negative = total_points - total_positive

    print(f"\nProcessing {len(image_items)} images...")
    print(f"  Total points: {total_points}")
    print(f"  Positive: {total_positive}, Ignored: {total_negative}")
    print("=" * 70)

    successful_images = 0
    failed_images = 0

    for idx, (image_path, data) in enumerate(image_items, 1):
        print(f"\n[{idx}/{len(image_items)}]")

        try:
            result = process_single_image(
                pipeline=pipeline,
                image_data=data,
                output_dir=output_dir,
                visualize_individual=visualize_individual,
                visualize_overview=visualize_overview,
            )

            all_results.append(result)

            if result["success"]:
                successful_images += 1
            else:
                failed_images += 1
                if not continue_on_failure:
                    print("Stopping processing due to failure")
                    break

        except Exception as e:
            failed_images += 1
            print(f"  ✗ Unexpected error: {e}")

            all_results.append(
                {
                    "image_path": image_path,
                    "image_name": f"failed_image_{idx}",
                    "num_points_total": len(data["points"]),
                    "num_positive": sum(data.get("is_background", [])),
                    "num_negative": len(data["points"]) - sum(data.get("is_background", [])),
                    "num_objects_segmented": 0,
                    "mean_score": 0,
                    "min_score": 0,
                    "max_score": 0,
                    "mean_iou": 0,
                    "min_iou": 0,
                    "max_iou": 0,
                    "total_time_sec": None,
                    "time_per_object_sec": None,
                    "success": False,
                    "error": str(e),
                }
            )

            if not continue_on_failure:
                break
            continue

    try:
        summary_df = pd.DataFrame(all_results)
        summary_path = Path(output_dir) / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Saved processing summary to {summary_path}")
    except Exception as e:
        print(f"\n✗ Failed to save processing summary: {e}")

    successful_results = [r for r in all_results if r["success"]]
    total_objects_segmented = sum(r["num_objects_segmented"] for r in all_results)

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"  Total images processed: {len(all_results)}")
    print(f"  Successful: {successful_images}, Failed: {failed_images}")
    if len(all_results) > 0:
        print(f"  Success rate: {successful_images / len(all_results) * 100:.1f}%")
    print(f"  Total positive objects segmented: {total_objects_segmented}")

    if successful_results:
        mean_scores = [r["mean_score"] for r in successful_results if r["num_objects_segmented"] > 0]
        mean_ious = [r["mean_iou"] for r in successful_results if r["num_objects_segmented"] > 0]
        if mean_scores:
            print(
                f"  Average confidence score (per image): "
                f"{np.mean(mean_scores):.3f} ± {np.std(mean_scores):.3f}"
            )
        if mean_ious:
            print(
                f"  Average IoU (per image):           "
                f"{np.mean(mean_ious):.3f} ± {np.std(mean_ious):.3f}"
            )

        times = [r["total_time_sec"] for r in successful_results if r.get("total_time_sec") is not None]
        times_per_obj = [
            r["time_per_object_sec"] for r in successful_results if r.get("time_per_object_sec") is not None
        ]
        if times:
            print(
                f"  Average processing time (per image): "
                f"{np.mean(times):.3f} s ± {np.std(times):.3f} s"
            )
        if times_per_obj:
            print(
                f"  Average time per object:            "
                f"{np.mean(times_per_obj):.3f} s ± {np.std(times_per_obj):.3f} s"
            )

    print(f"\nResults saved to: {output_dir}/")
    if successful_images > 0:
        print(f"  - Masks: {output_dir}/masks/")
        print(f"  - Visualizations: {output_dir}/visualizations/")
        print(f"  - Metadata: {output_dir}/metadata/")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    print("Initializing SAM 3 pipeline with exemplar-first text prompts + box fallback...")

    exemplar_bank = load_exemplar_bank(EXEMPLAR_BANK_MODE, EXEMPLAR_BANK_PATH)

    try:
        pipeline = UnderwaterSAM3Pipeline(
            model_id="facebook/sam3",
            device="cpu",
            exemplar_bank=exemplar_bank,
        )
    except Exception as e:
        print(f"✗ Failed to initialize SAM 3 pipeline: {e}")
        raise SystemExit(1)

    print("\nLoading annotations...")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"✓ Loaded {len(df)} annotations")

        if "background" not in df.columns:
            print("⚠ Warning: 'background' column not found, treating all as positive")
            df["background"] = 1

        if "label_lineage_names" not in df.columns:
            raise KeyError("Column 'label_lineage_names' not found in CSV")

        df["label_lineage_names"] = df["label_lineage_names"].astype(str)

        # Optional example filter: keep only rows with "Cup" in lineage
        mask_eck = df["label_lineage_names"].str.contains("Cup", case=False, na=False)
        eck_image_paths = df.loc[mask_eck, "point_media_path_best"].unique()
        print(f"✓ Found {len(eck_image_paths)} images with 'Cup' in lineage")

        df = df[df["point_media_path_best"].isin(eck_image_paths)].copy()
        print(f"✓ Filtered to {len(df)} annotations on 'Cup' images")

        num_positive = len(df[df["background"] == 1])
        num_negative = len(df[df["background"] == 0])
        print(f"  Positive points (background==1): {num_positive}")
        print(f"  Negative/ignored points (background==0): {num_negative}")

        print("\nFirst few filtered rows:")
        cols_to_show = [
            c
            for c in ["label_name", "background", "point_polygon", "label_lineage_names"]
            if c in df.columns
        ]
        print(df[cols_to_show].head(5).to_string())

    except Exception as e:
        print(f"✗ Failed to load annotations: {e}")
        print("Make sure your CSV file has these columns:")
        print("  - point_media_path_best (image URL or path)")
        print("  - point_x, point_y (normalized 0-1)")
        print("  - label_name (object label)")
        print("  - background (1=positive/use, 0=ignore)")
        print("  - label_translated_lineage_names (for filtering & prompts)")
        print("  - point_polygon (normalized polygon coordinates for IoU)")
        raise SystemExit(1)

    print("\nPreparing annotations...")
    try:
        image_annotations = prepare_annotations_from_df(df)
        print(f"✓ Prepared annotations for {len(image_annotations)} unique images")

        total_objects = sum(len(data["points"]) for data in image_annotations.values())
        print(f"✓ Total objects (rows) to process: {total_objects}")

    except Exception as e:
        print(f"✗ Failed to prepare annotations: {e}")
        raise SystemExit(1)

    print("\nStarting dataset processing with exemplar-first SAM3 pipeline (Option A)...")
    print("=" * 70)
    print("IMPORTANT: Change max_images=None in process_dataset to process full dataset")
    print("=" * 70)

    try:
        results = process_dataset(
            pipeline=pipeline,
            image_annotations=image_annotations,
            output_dir=OUTPUT_DIR,
            max_images=None,  # set to e.g. 10 for debugging
            visualize_individual=True,
            visualize_overview=True,
            continue_on_failure=True,
        )

        print("\n✓ Pipeline complete!")
        print("\nNext steps:")
        print(f"  1. Check {OUTPUT_DIR}/visualizations/ for segmentation quality")
        print(f"  2. Check {OUTPUT_DIR}/metadata/ for per-object scores and IoU")
        print(f"  3. Inspect {OUTPUT_DIR}/processing_summary.csv for per-image metrics")

        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user (Ctrl+C)")
        print(f"Partial results may be saved in {OUTPUT_DIR}/")

    except Exception as e:
        print(f"\n✗ Pipeline failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
