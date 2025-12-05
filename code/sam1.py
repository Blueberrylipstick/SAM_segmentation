import torch
import numpy as np
import cv2
import ast
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import time

from segment_anything import sam_model_registry, SamPredictor


# ================================================================
# Utility: Compute IoU
# ================================================================
def compute_iou(mask_pred, mask_gt):
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)

    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()

    if union == 0:
        return 0.0
    return intersection / union


# ================================================================
# Convert relative polygon offsets to GT mask
# ================================================================
def polygon_to_mask(relative_polygon, cx, cy, height, width):
    try:
        if isinstance(relative_polygon, str):
            rel_poly = ast.literal_eval(relative_polygon)
        else:
            rel_poly = relative_polygon
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)

    abs_poly = []
    for dx, dy in rel_poly:
        ax = np.clip(cx + dx, 0, 1)
        ay = np.clip(cy + dy, 0, 1)
        px = int(ax * width)
        py = int(ay * height)
        abs_poly.append([px, py])

    if len(abs_poly) < 3:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(abs_poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


# ================================================================
# SAM1 Pipeline
# ================================================================
class UnderwaterSAMPipeline:
    def __init__(self, checkpoint_path, model_type="vit_b", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading SAM1 model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)
        print(f"✓ SAM1 initialized on {self.device}")

    # ------------------------------------------------------------
    # Image loader (URL & local)
    # ------------------------------------------------------------
    def load_image(self, image_path):
        image_path = str(image_path)

        if image_path.startswith("http"):
            try:
                resp = requests.get(
                    image_path,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                resp.raise_for_status()
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode image")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise ValueError(f"Failed to load URL image: {e}")

        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Local image not found: {image_path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"OpenCV failed to load: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------
    # Predict mask using positive + negative points
    # ------------------------------------------------------------
    def predict_mask(self, image, cx, cy, negative_points):
        H, W = image.shape[:2]

        px = int(cx * W)
        py = int(cy * H)

        coords = [[px, py]]
        labels = [1]  # positive

        for nx, ny in negative_points:
            coords.append([nx, ny])
            labels.append(0)

        coords = np.array(coords)
        labels = np.array(labels)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=False  # we want 1 mask
        )

        return masks[0], float(scores[0])

    # ------------------------------------------------------------
    # Process one object
    # ------------------------------------------------------------
    def process_object(self, image, cx, cy, label, polygon, neg_pxpy):
        H, W = image.shape[:2]

        gt_mask = polygon_to_mask(polygon, cx, cy, H, W)
        pred_mask, score = self.predict_mask(image, cx, cy, neg_pxpy)

        iou = compute_iou(pred_mask > 0.5, gt_mask)

        return {
            "label": label,
            "cx": cx,
            "cy": cy,
            "mask_pred": pred_mask,
            "mask_gt": gt_mask,
            "iou": iou,
            "score": score
        }

    # ------------------------------------------------------------
    # Visualize single object
    # ------------------------------------------------------------
    def visualize_object(self, image, result, save_path=None):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(result["mask_gt"], cmap="jet")
        ax[1].set_title("GT Mask")
        ax[1].axis("off")

        ax[2].imshow(result["mask_pred"], cmap="viridis")
        ax[2].set_title(
            f"Predicted\nIoU: {result['iou']:.3f}\nScore: {result['score']:.3f}"
        )
        ax[2].axis("off")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ================================================================
# Annotation preparation
# ================================================================
def prepare_annotations_from_df(df):
    ann = defaultdict(lambda: {"objects": []})
    for _, row in df.iterrows():
        try:
            img = row["point_media_path_best"]
            cx = float(row["point_x"])
            cy = float(row["point_y"])
            label = str(row.get("label_name", "object"))
            polygon = row["point_polygon"]
            bg = int(row.get("background", 1))

            ann[img]["objects"].append({
                "cx": cx,
                "cy": cy,
                "label": label,
                "polygon": polygon,
                "is_bg": bg
            })
        except:
            pass
    return dict(ann)


# ================================================================
# Save masks + metadata
# ================================================================
def save_masks(result, name, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{name}_pred.npy", result["mask_pred"])
    np.save(outdir / f"{name}_gt.npy", result["mask_gt"])


# Updated: accepts optional per-object time
def save_metadata(results, image_name, image_path, out_csv, time_per_object=None):
    rows = []
    for idx, r in enumerate(results):
        row = {
            "image_path": image_path,
            "object_index": idx,
            "label": r["label"],
            "cx": r["cx"],
            "cy": r["cy"],
            "score": r["score"],
            "iou": r["iou"]
        }
        # add the same per-object time for all objects of this image
        if time_per_object is not None:
            row["time_per_object_sec"] = time_per_object

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ================================================================
# Process single image
# ================================================================
def process_single_image(pipeline, image_path, objects, outdir):
    print(f"\nProcessing: {image_path}")

    # --- START TIMER ---
    start_time = time.perf_counter()

    try:
        image = pipeline.load_image(image_path)
    except Exception as e:
        print("  ✗ Failed to load image:", e)
        return None

    # ❗ FIX — set SAM1 image embedding
    try:
        pipeline.predictor.set_image(image)
    except Exception as e:
        print("  ✗ Failed to set image in SAM1 predictor:", e)
        return None

    H, W = image.shape[:2]

    # negative points = all background=0 rows
    negative_points = [
        (int(o["cx"] * W), int(o["cy"] * H))
        for o in objects
        if o["is_bg"] == 0
    ]

    results = []
    for o in objects:
        if o["is_bg"] != 1:
            continue

        r = pipeline.process_object(
            image=image,
            cx=o["cx"],
            cy=o["cy"],
            label=o["label"],
            polygon=o["polygon"],
            neg_pxpy=negative_points
        )
        results.append(r)

    print(f"  ✓ {len(results)} objects processed")

    img_name = Path(image_path).stem
    mask_dir = Path(outdir) / "masks" / img_name
    vis_dir = Path(outdir) / "visualizations" / img_name
    meta_dir = Path(outdir) / "metadata"

    for idx, r in enumerate(results):
        save_masks(r, f"{img_name}_obj{idx}", mask_dir)
        pipeline.visualize_object(
            image,
            r,
            save_path=vis_dir / f"{img_name}_obj{idx}.png"
        )

    # --- STOP TIMER & COMPUTE TIMES ---
    end_time = time.perf_counter()
    total_time = end_time - start_time
    num_objects = len(results)
    time_per_object = total_time / num_objects if num_objects else 0.0

    print(f"  Total time: {total_time:.3f} s "
          f"({time_per_object:.3f} s per object)")

    # pass time_per_object into metadata
    df_meta = save_metadata(
        results,
        img_name,
        image_path,
        meta_dir / f"{img_name}.csv",
        time_per_object=time_per_object,
    )

    mean_iou = df_meta["iou"].mean() if len(df_meta) else 0
    print(f"  Mean IoU: {mean_iou:.3f}")

    return {
        "image_path": image_path,
        "num_objects": num_objects,
        "mean_iou": mean_iou,
        "total_time_sec": total_time,
        "time_per_object_sec": time_per_object
    }


# ================================================================
# Process entire dataset
# ================================================================
def process_dataset(pipeline, annotations, outdir, max_images=None):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    items = list(annotations.items())
    if max_images:
        items = items[:max_images]

    summary = []
    for i, (img, data) in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}]")
        r = process_single_image(pipeline, img, data["objects"], outdir)
        if r:
            summary.append(r)

    df = pd.DataFrame(summary)
    df.to_csv(Path(outdir) / "processing_summary.csv", index=False)
    print("\n======================================")
    print("          DATASET COMPLETE")
    print("======================================")
    print(f"Average IoU = {df['mean_iou'].mean():.3f}")

    return df


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    csv_path = "../data/annotations-u45-Flinders_Western_Boundary_2011_targeted-Flinders_Western_Boundary_2011_targeted-copy-18164-35952eb26deb642336fe-dataframe.csv"

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["point_polygon", "point_x", "point_y", "point_media_path_best"])
    if "background" not in df.columns:
        df["background"] = 1

    annotations = prepare_annotations_from_df(df)

    pipeline = UnderwaterSAMPipeline(
        checkpoint_path="work_dir/SAM/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu"
    )

    summary = process_dataset(
        pipeline,
        annotations,
        outdir="results_sam1_iou",
        max_images=None
    )

    print("\nDONE. Mean IoU:", summary["mean_iou"].mean())
