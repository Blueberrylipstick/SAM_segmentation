import ast
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


# ==============================
# 1. GEOMETRY / IMAGE HELPERS
# ==============================

def polygon_to_mask(relative_polygon, center_norm, height, width):
    """
    Convert a polygon given as *relative normalized offsets* around the point
    center into a binary mask.

    relative_polygon: list [[dx, dy], ...] with values ~[-0.2, 0.2]
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

        if len(poly_px) == 0:
            return None

        poly_px = np.array(poly_px, dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_px], 1)

        return mask.astype(bool)

    except Exception as e:
        print(f"Error converting polygon to mask: {e}")
        return None


def load_image(image_path, max_retries=3, timeout=20):
    """
    Robust image loader: local path or HTTP(S) URL.
    Returns RGB np.array.
    """
    if isinstance(image_path, str) and image_path.startswith("http"):
        for attempt in range(max_retries):
            try:
                print(f"    Downloading URL (attempt {attempt+1}/{max_retries})...")
                r = requests.get(
                    image_path,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                r.raise_for_status()
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode image bytes")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except Exception as e:
                print(f"    ✗ URL attempt failed: {e}")
                if attempt == max_retries - 1:
                    raise
        return None
    else:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file does not exist: {image_path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read local image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute tight bounding box (x1, y1, x2, y2) for a binary mask.
    Returns None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


def extract_patch(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the image to the tight bounding box of mask.
    Returns the cropped patch (RGB); or None if mask empty.
    """
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox

    return image[y1:y2+1, x1:x2+1, :]


# ==============================
# 2. QUALITY METRICS (Option D)
# ==============================

def compute_texture_score(patch: np.ndarray) -> float:
    """
    Simple texture / edge strength metric:
      - convert to grayscale
      - Laplacian variance + intensity std
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    intensity_std = float(gray.std())
    return lap_var + intensity_std


def normalize_list(values: List[float]) -> List[float]:
    """
    Min-max normalize list to [0,1].
    If all values equal or list empty → 0.5 for all.
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return [0.5] * len(values)
    norm = (arr - vmin) / (vmax - vmin)
    return norm.tolist()


def compute_quality_scores_for_label(exemplars: List[Dict[str, Any]]) -> List[float]:
    """
    Given a list of exemplars (all from the SAME label),
    compute combined quality score (Option D):

      quality = 0.4 * area_norm
              + 0.4 * center_overlap_norm
              + 0.2 * texture_norm

    center_overlap is already in {0,1}, but we still min-max it
    across exemplars of the same label (so if all are 1 → all 0.5).
    """
    areas = [ex["mask_area"] for ex in exemplars]
    centers = [ex["center_overlap"] for ex in exemplars]
    textures = [ex["texture_score"] for ex in exemplars]

    area_n = normalize_list(areas)
    center_n = normalize_list(centers)
    texture_n = normalize_list(textures)

    qualities = []
    for a, c, t in zip(area_n, center_n, texture_n):
        q = 0.4 * a + 0.4 * c + 0.2 * t
        qualities.append(float(q))

    return qualities


# ==============================
# 3. BUILD RAW EXEMPLARS
# ==============================

def get_leaf_label(x: Any) -> str:
    parts = [p.strip() for p in str(x).split(">")]
    return parts[-1] if parts else None


def build_raw_exemplar_bank(
    csv_path: str,
    require_background_positive: bool = True,
    min_mask_area_frac: float = 1e-4,
    max_mask_area_frac: float = 0.8
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a RAW exemplar bank:
    - key: leaf label (last level of label_lineage_names)
    - value: list of exemplar dicts for that label
    """

    df = pd.read_csv(csv_path, low_memory=False)

    if "label_lineage_names" not in df.columns:
        raise KeyError("CSV must contain 'label_lineage_names' column")

    df["leaf_label"] = df["label_lineage_names"].apply(get_leaf_label)

    if require_background_positive and "background" in df.columns:
        df = df[df["background"] == 1]

    mask_valid = (
        df["point_polygon"].notna()
        & df["point_x"].notna()
        & df["point_y"].notna()
        & df["point_media_path_best"].notna()
        & df["leaf_label"].notna()
    )
    df_valid = df[mask_valid].copy()

    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid candidate rows: {len(df_valid)}")

    raw_bank: Dict[str, List[Dict[str, Any]]] = {}
    failed_rows = []

    grouped = df_valid.groupby("leaf_label", sort=False)

    for leaf_label, group in grouped:
        print(f"\n=== Processing label: {leaf_label} (rows: {len(group)}) ===")

        for idx, row in group.iterrows():
            try:
                image_path = str(row["point_media_path_best"])
                point_x = float(row["point_x"])
                point_y = float(row["point_y"])
                polygon = row["point_polygon"]
                full_lineage = row["label_lineage_names"]
                simple_label = row.get("label_name", leaf_label)

                image = load_image(image_path)
                h, w = image.shape[:2]
                img_area = float(h * w)

                center_norm = [point_x, point_y]
                mask = polygon_to_mask(polygon, center_norm, h, w)

                if mask is None:
                    print(f"  - row {idx}: mask is None → skip")
                    failed_rows.append(idx)
                    continue

                mask_area = float(mask.sum())
                if mask_area <= 0:
                    print(f"  - row {idx}: empty mask → skip")
                    failed_rows.append(idx)
                    continue

                area_frac = mask_area / img_area
                if area_frac < min_mask_area_frac:
                    print(f"  - row {idx}: mask too small (frac={area_frac:.2e}) → skip")
                    failed_rows.append(idx)
                    continue
                if area_frac > max_mask_area_frac:
                    print(f"  - row {idx}: mask huge (frac={area_frac:.3f}), keeping but flagged")

                px = int(point_x * w)
                py = int(point_y * h)
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))

                center_overlap = 1.0 if mask[py, px] > 0 else 0.0

                if center_overlap == 0.0:
                    # "Repair": move center_norm to mask centroid
                    ys, xs = np.where(mask > 0)
                    cx = float(xs.mean()) / w
                    cy = float(ys.mean()) / h
                    center_norm = [cx, cy]
                    print(f"  - row {idx}: center not in mask, repaired to centroid ({cx:.3f},{cy:.3f})")

                # Extract patch for texture analysis
                patch = extract_patch(image, mask)
                if patch is None or patch.size == 0:
                    print(f"  - row {idx}: patch empty → skip")
                    failed_rows.append(idx)
                    continue

                texture_score = compute_texture_score(patch)

                exemplar = {
                    "image": image,
                    "mask": mask.astype(np.uint8),
                    "image_path": image_path,
                    "full_lineage": full_lineage,
                    "leaf_label": leaf_label,
                    "label_name": simple_label,
                    "center_norm": center_norm,
                    "mask_area": mask_area,
                    "area_frac": area_frac,
                    "center_overlap": center_overlap,
                    "texture_score": texture_score,
                    "csv_row_index": int(idx),
                }

                raw_bank.setdefault(leaf_label, []).append(exemplar)
                print(f"  ✓ row {idx}: exemplar added (area_frac={area_frac:.4f}, "
                      f"center_overlap={center_overlap:.1f}, texture={texture_score:.2f})")

            except Exception as e:
                print(f"  ✗ row {idx}: error building exemplar: {e}")
                failed_rows.append(idx)
                continue

    print("\n===================================")
    print(f"RAW exemplar bank: {len(raw_bank)} labels with at least 1 exemplar")
    if failed_rows:
        print(f"Failed rows ({len(failed_rows)}). First few: {failed_rows[:10]}")

    return raw_bank


# ==============================
# 4. SELECT BEST EXEMPLAR / LABEL
# ==============================

def select_best_exemplar_per_label(
    raw_bank: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Given the raw bank (multiple exemplars per label),
    compute quality scores and keep only the best exemplar for each label.
    """
    best_bank: Dict[str, Dict[str, Any]] = {}

    for label, exemplars in raw_bank.items():
        if not exemplars:
            continue

        qualities = compute_quality_scores_for_label(exemplars)

        for ex, q in zip(exemplars, qualities):
            ex["quality_score"] = q

        best_idx = int(np.argmax(qualities))
        best_ex = exemplars[best_idx]

        print(f"Label '{label}': {len(exemplars)} exemplars, best idx={best_idx}, "
              f"quality={best_ex['quality_score']:.3f}, "
              f"area_frac={best_ex['area_frac']:.4f}, "
              f"center_overlap={best_ex['center_overlap']:.1f}")

        best_bank[label] = best_ex

    print("\n===================================")
    print(f"BEST exemplar bank: {len(best_bank)} labels")

    return best_bank


# ==============================
# 5. VISUALIZATION UTILITIES
# ==============================

def visualize_single_exemplar(ex: Dict[str, Any], title: str = None, show: bool = True,
                              save_path: str = None):
    """
    Visualize a single exemplar: full image + mask overlay + cropped patch.
    """
    image = ex["image"]
    mask = ex["mask"].astype(bool)

    patch = extract_patch(image, mask)
    if patch is None:
        patch = np.zeros((50, 50, 3), dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Full image")
    axes[0].axis("off")

    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 0] = 255
    alpha = 0.4
    overlay[mask] = ((1 - alpha) * image[mask] + alpha * red[mask]).astype(np.uint8)

    axes[1].imshow(overlay)
    axes[1].set_title("Mask overlay")
    axes[1].axis("off")

    # 3) patch
    axes[2].imshow(patch)
    axes[2].set_title("Cropped patch")
    axes[2].axis("off")

    if title is None:
        title = f"{ex.get('leaf_label', '')} | score={ex.get('quality_score', 0):.3f}"
    fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved exemplar visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_exemplar_grid(
    best_bank: Dict[str, Dict[str, Any]],
    output_path: str = "exemplar_grid.png",
    max_labels: int = 40,
    ncols: int = 5
):
    """
    Export a grid of exemplar patches (best per label) for reporting.
    """
    labels = list(best_bank.keys())[:max_labels]
    n = len(labels)
    if n == 0:
        print("No exemplars to export grid for.")
        return

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, label in enumerate(labels):
        ax = axes[i]
        ex = best_bank[label]
        image = ex["image"]
        mask = ex["mask"].astype(bool)
        patch = extract_patch(image, mask)
        if patch is None:
            patch = np.zeros((50, 50, 3), dtype=np.uint8)

        ax.imshow(patch)
        q = ex.get("quality_score", 0.0)
        ax.set_title(f"{label}\nq={q:.3f}", fontsize=9)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Exemplar grid saved to {out_path}")
    plt.close(fig)


# ==============================
# 6. MAIN ORCHESTRATOR
# ==============================

def build_and_save_exemplar_banks(
    csv_path: str,
    raw_output_path: str = "leaf_exemplar_bank_raw.pkl",
    best_output_path: str = "leaf_exemplar_bank_best.pkl",
    grid_output_path: str = "leaf_exemplar_grid.png"
):
    """
    High-level function:
      1) build raw exemplars
      2) select best exemplar per label
      3) save both banks
      4) export a grid preview of best exemplars
    """
    csv_path = str(csv_path)
    print(f"=== Building exemplar banks from {csv_path} ===")

    raw_bank = build_raw_exemplar_bank(csv_path)
    best_bank = select_best_exemplar_per_label(raw_bank)

    raw_output_path = Path(raw_output_path)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_output_path, "wb") as f:
        pickle.dump(raw_bank, f)
    print(f"\nRAW exemplar bank saved to: {raw_output_path}")

    best_output_path = Path(best_output_path)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_output_path, "wb") as f:
        pickle.dump(best_bank, f)
    print(f"BEST exemplar bank saved to: {best_output_path}")

    export_exemplar_grid(best_bank, output_path=grid_output_path)

    print("\nDone building exemplar banks.")

    return raw_bank, best_bank


if __name__ == "__main__":
    csv_path = "/Users/yulch/Desktop/sam/annotations_u45_Flinders_Western_Boundary_2011_targeted_Flinders.csv"

    raw_bank_path = "leaf_exemplar_bank_raw.pkl"
    best_bank_path = "leaf_exemplar_bank_best.pkl"
    grid_path = "leaf_exemplar_grid.png"

    build_and_save_exemplar_banks(
        csv_path,
        raw_output_path=raw_bank_path,
        best_output_path=best_bank_path,
        grid_output_path=grid_path,
    )
