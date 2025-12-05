import torch
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import requests
import ast

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def polygon_to_mask(relative_polygon, center_norm, height, width):
    """
    Convert a polygon given as *relative normalized offsets* around the point
    center into a binary mask.

    relative_polygon: list [[dx, dy], ...] with values ~[-0.2, 0.2]
    center_norm: [cx, cy] in [0,1] (this is point_x, point_y)
    """
    try:
        # If polygon stored as a string, parse it
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

    # SAM2 masks can be float; threshold them
    pred = (mask_pred > 0.5)
    gt = mask_gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


class UnderwaterSAM2Pipeline:
    def __init__(self, checkpoint_path, model_cfg="sam2_hiera_s.yaml", device='cpu'):
        """
        Initialize SAM 2 for underwater image segmentation

        Args:
            checkpoint_path: Path to SAM 2 checkpoint (.pt file)
            model_cfg: Model configuration (sam2_hiera_t.yaml, sam2_hiera_s.yaml, etc.)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        print(f"Loading SAM 2 model from {checkpoint_path}...")
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print(f"✓ SAM 2 initialized on {device}")

    def load_image(self, image_path, max_retries=3, timeout=30):
        """
        Load image from URL or local path with robust error handling

        Args:
            image_path: Path or URL to image
            max_retries: Number of retry attempts for URL downloads
            timeout: Timeout in seconds for URL requests

        Returns:
            image: RGB image array

        Raises:
            ValueError: If image cannot be loaded after all attempts
        """
        if image_path.startswith('http'):
            for attempt in range(max_retries):
                try:
                    print(f"    Downloading from URL (attempt {attempt + 1}/{max_retries})...")
                    response = requests.get(
                        image_path, timeout=timeout,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    response.raise_for_status()

                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if image is None:
                        raise ValueError("Failed to decode image data")

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"    ✓ Successfully loaded image from URL")
                    return image

                except requests.exceptions.RequestException as e:
                    print(f"    ✗ Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to download image after {max_retries} attempts: {e}")
                    continue

                except Exception as e:
                    print(f"    ✗ Attempt {attempt + 1} failed with error: {e}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to process downloaded image: {e}")
                    continue
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

    def segment_image_per_object_with_labels(
        self,
        image_path,
        normalized_points,
        label_names=None,
        is_background=None,
        polygons=None,
        multimask_output=False
    ):
        """
        Segment objects using positive points only (based on background column).

        IMPORTANT:
        - background == 1  → POSITIVE point, included in segmentation
        - background == 0  → NEGATIVE / ignore point, NOT used in segmentation

        Args:
            image_path: Path or URL to image
            normalized_points: List of [x, y] normalized coordinates (0-1)
            label_names: List of label names for each point
            is_background: List of 0/1 flags (1=positive/use, 0=ignore)
            polygons: List of polygons (normalized 0-1) for IoU computation
            multimask_output: If True, returns 3 candidate masks per point

        Returns:
            results: List of dicts for POSITIVE objects only
            image: Loaded image array
        """
        try:
            image = self.load_image(image_path)
            height, width = image.shape[:2]
            print(f"    Image dimensions: {width}x{height}")
        except Exception as e:
            raise ValueError(f"Image loading failed: {e}")

        try:
            self.predictor.set_image(image)
        except Exception as e:
            raise ValueError(f"Failed to set image in predictor: {e}")

        # If no mask usage info provided, treat all as positive/use
        if is_background is None:
            is_background = [1] * len(normalized_points)

        # Collect only positive points (background == 1 means "use this point")
        foreground_points = []
        foreground_labels = []
        foreground_indices = []

        for idx, (norm_point, use_flag) in enumerate(zip(normalized_points, is_background)):
            if use_flag != 1:
                # Skip points where background == 0 (negative / not used)
                continue

            if not (0 <= norm_point[0] <= 1 and 0 <= norm_point[1] <= 1):
                continue

            foreground_points.append(norm_point)
            foreground_labels.append(label_names[idx] if label_names else f"Object_{idx}")
            foreground_indices.append(idx)

        print(f"    Positive points used for segmentation: {len(foreground_points)}")

        results = []
        failed_objects = []

        for list_idx, (norm_point, label, orig_idx) in enumerate(
            zip(foreground_points, foreground_labels, foreground_indices)
        ):
            try:
                pixel_x = int(norm_point[0] * width)
                pixel_y = int(norm_point[1] * height)

                # Only a single positive point per object
                all_points = np.array([[pixel_x, pixel_y]])
                all_labels = np.array([1])

                masks, scores, logits = self.predictor.predict(
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=multimask_output
                )

                if len(masks) == 0 or masks[0] is None:
                    print(f"    ⚠ Warning: No valid mask for object {orig_idx}, skipping")
                    failed_objects.append(orig_idx)
                    continue

                # Build GT mask for this object (if polygon is available)
                gt_mask = None
                if polygons is not None and orig_idx < len(polygons):
                    gt_poly = polygons[orig_idx]
                    # norm_point is the [cx, cy] for this object
                    gt_mask = polygon_to_mask(gt_poly, norm_point, height, width)

                # Compute IoU between SAM2 mask and GT mask
                iou_value = compute_iou(masks[0], gt_mask) if gt_mask is not None else 0.0

                results.append({
                    'mask': masks[0],
                    'score': float(scores[0]),
                    'iou': float(iou_value),
                    'all_masks': masks if multimask_output else None,
                    'all_scores': scores.tolist() if multimask_output else None,
                    'normalized_point': norm_point,
                    'pixel_point': [pixel_x, pixel_y],
                    'label_name': label,
                    'object_id': orig_idx,
                    'is_background': False,  # we are only storing positive objects here
                    'success': True
                })

            except Exception as e:
                print(f"    ✗ Failed to process foreground object {orig_idx} ({label}): {e}")
                failed_objects.append(orig_idx)
                continue

        if failed_objects:
            print(f"    ⚠ {len(failed_objects)} objects failed: {failed_objects}")

        if len(results) == 0:
            print("    ⚠ Warning: No positive objects were successfully segmented")

        return results, image

    def visualize_individual_results(self, image, results, save_dir=None,
                                     image_name="image", show=False):
        """
        Visualize each object's segmentation individually,
        including SAM score and IoU w.r.t. GT.
        """
        saved_paths = []
        failed_visualizations = []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for result in results:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                axes[0].imshow(image)
                axes[0].scatter(
                    result['pixel_point'][0], result['pixel_point'][1],
                    c='red', s=300, marker='*', edgecolors='white', linewidths=2
                )
                axes[0].set_title(f"Object: {result['label_name']}\nPoint: {result['pixel_point']}")
                axes[0].axis('off')

                axes[1].imshow(image)
                axes[1].imshow(result['mask'], alpha=0.5, cmap='jet')
                axes[1].set_title(
                    f"Segmentation\nScore: {result['score']:.3f}, IoU: {result.get('iou', 0.0):.3f}"
                )
                axes[1].axis('off')

                plt.tight_layout()

                if save_dir:
                    safe_label = result['label_name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    save_path = Path(save_dir) / f"{image_name}_obj{result['object_id']:03d}_{safe_label}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    saved_paths.append(save_path)

                if show:
                    plt.show()
                else:
                    plt.close()

            except Exception as e:
                print(f"    ✗ Failed to create visualization for object {result['object_id']}: {e}")
                failed_visualizations.append(result['object_id'])
                if 'fig' in locals():
                    plt.close(fig)
                continue

        if failed_visualizations:
            print(f"    ⚠ Failed to create {len(failed_visualizations)} visualizations")

        return saved_paths

    def visualize_all_objects(self, image, results, save_path=None, show=False):
        """
        Visualize all objects in one comprehensive figure, with score + IoU.
        """
        try:
            num_objects = len(results)
            if num_objects == 0:
                print("    ⚠ No results to visualize")
                return None

            cols = min(4, num_objects + 1)
            rows = max(1, (num_objects + cols) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            # Original image with points
            axes[0].imshow(image)
            for result in results:
                axes[0].scatter(
                    result['pixel_point'][0], result['pixel_point'][1],
                    c='red', s=200, marker='*', edgecolors='white', linewidths=2
                )
            axes[0].set_title(f'Original Image\n{num_objects} objects')
            axes[0].axis('off')

            # Each object with its mask
            for idx, result in enumerate(results):
                if idx + 1 < len(axes):
                    ax = axes[idx + 1]
                    ax.imshow(image)
                    ax.imshow(result['mask'], alpha=0.5, cmap='jet')
                    ax.set_title(
                        f"{result['label_name']}\nScore: {result['score']:.3f}, IoU: {result.get('iou', 0.0):.3f}",
                        fontsize=10
                    )
                    ax.axis('off')

            # Hide unused subplots
            for idx in range(num_objects + 1, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"    ✓ Saved overview to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig

        except Exception as e:
            print(f"    ✗ Failed to create overview visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None


def prepare_annotations_from_df(df):
    """
    Convert annotation DataFrame to grouped format with positive/negative flags and polygons.

    Assumptions:
    - point_media_path_best: image path
    - point_x, point_y: normalized center coordinates [0,1]
    - label_name: object label
    - background: 1 = positive/use, 0 = negative/ignore
    - point_polygon: list of normalized polygon points [[x,y], ...]
    """
    image_annotations = defaultdict(lambda: {
        'points': [],
        'label_names': [],
        'is_background': [],   # we use 1=positive/use, 0=ignore
        'polygons': [],
        'image_path': None
    })

    invalid_rows = []

    for idx, row in df.iterrows():
        try:
            if pd.isna(row['point_media_path_best']) or pd.isna(row['point_x']) or pd.isna(row['point_y']):
                invalid_rows.append(idx)
                continue

            image_path = str(row['point_media_path_best'])
            point_x = float(row['point_x'])
            point_y = float(row['point_y'])
            label_name = str(row.get('label_name', 'Unknown'))
            use_flag = int(row.get('background', 1))  # 1 = positive/use, 0 = ignore

            if not (0 <= point_x <= 1 and 0 <= point_y <= 1):
                print(f"⚠ Warning: Invalid coordinates ({point_x}, {point_y}) for row {idx}, skipping")
                invalid_rows.append(idx)
                continue

            image_annotations[image_path]['points'].append([point_x, point_y])
            image_annotations[image_path]['label_names'].append(label_name)
            image_annotations[image_path]['is_background'].append(use_flag)
            image_annotations[image_path]['polygons'].append(row['point_polygon'])
            image_annotations[image_path]['image_path'] = image_path

        except Exception as e:
            print(f"✗ Error processing row {idx}: {e}")
            invalid_rows.append(idx)
            continue

    if invalid_rows:
        print(f"⚠ Skipped {len(invalid_rows)} invalid rows")

    return dict(image_annotations)


def save_individual_masks(results, output_dir, image_name):
    """
    Save each object's mask separately as .npy file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []
    failed_saves = []

    for result in results:
        try:
            safe_label = result['label_name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
            mask_path = Path(output_dir) / f"{image_name}_obj{result['object_id']:03d}_{safe_label}_mask.npy"
            np.save(mask_path, result['mask'])
            saved_paths.append(mask_path)
        except Exception as e:
            print(f"    ✗ Failed to save mask for object {result['object_id']}: {e}")
            failed_saves.append(result['object_id'])
            continue

    if failed_saves:
        print(f"    ⚠ Failed to save {len(failed_saves)} masks")

    return saved_paths


def save_results_metadata(results, output_path, image_name, image_path, time_per_object_sec=None):
    """
    Save metadata about segmentation results to CSV, including IoU.
    Optionally add per-object processing time (same for all objects of this image).
    """
    try:
        metadata = []

        for result in results:
            row = {
                'image_name': image_name,
                'image_path': image_path,
                'object_id': result['object_id'],
                'label_name': result['label_name'],
                'point_x_normalized': result['normalized_point'][0],
                'point_y_normalized': result['normalized_point'][1],
                'point_x_pixel': result['pixel_point'][0],
                'point_y_pixel': result['pixel_point'][1],
                'segmentation_score': result['score'],
                'gt_iou': result.get('iou', 0.0),
                'mask_shape': str(result['mask'].shape),
                'is_background': result.get('is_background', False),
                'success': result.get('success', True)
            }
            if time_per_object_sec is not None:
                row['time_per_object_sec'] = time_per_object_sec

            metadata.append(row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)

        return df

    except Exception as e:
        print(f"    ✗ Failed to save metadata: {e}")
        return None


def process_single_image(pipeline, image_data, output_dir='results_evaluation',
                         visualize_individual=True, visualize_overview=True):
    """
    Process a single image with positive point labels based on `background` column.

    background == 1 → positive point used in segmentation
    background == 0 → negative / ignored
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_path = image_data['image_path']
    image_name = Path(image_path).stem if not image_path.startswith('http') else f"image_{abs(hash(image_path)) % 100000}"

    total_points = len(image_data['points'])
    use_flags = image_data.get('is_background', [1] * total_points)
    num_positive = sum(use_flags)
    num_negative = total_points - num_positive

    print(f"Processing: {image_name}")
    print(f"  Total points: {total_points} (Positive: {num_positive}, Ignored: {num_negative})")

    # start per-image timer
    start_time = time.perf_counter()

    try:
        results, image = pipeline.segment_image_per_object_with_labels(
            image_path=image_path,
            normalized_points=image_data['points'],
            label_names=image_data['label_names'],
            is_background=image_data.get('is_background'),
            polygons=image_data.get('polygons'),
            multimask_output=False
        )

        print(f"  ✓ Successfully segmented {len(results)} positive objects")

        masks_dir = Path(output_dir) / 'masks'
        vis_dir = Path(output_dir) / 'visualizations'
        metadata_dir = Path(output_dir) / 'metadata'

        # Save masks
        try:
            mask_paths = save_individual_masks(results, masks_dir, image_name)
            print(f"  ✓ Saved {len(mask_paths)} masks")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to save some masks: {e}")

        # Visualizations
        if visualize_individual:
            try:
                vis_paths = pipeline.visualize_individual_results(
                    image=image,
                    results=results,
                    save_dir=vis_dir / 'individual',
                    image_name=image_name,
                    show=False
                )
                print(f"  ✓ Created {len(vis_paths)} visualizations")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to create visualizations: {e}")

        if visualize_overview:
            try:
                overview_path = vis_dir / 'overview' / f"{image_name}_overview.png"
                pipeline.visualize_all_objects(
                    image=image,
                    results=results,
                    save_path=overview_path,
                    show=False
                )
                print(f"  ✓ Created overview")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to create overview: {e}")

        # Compute scores & IoUs
        scores = [r['score'] for r in results]
        ious = [r.get('iou', 0.0) for r in results]

        # stop timer and compute per-image times
        end_time = time.perf_counter()
        total_time = end_time - start_time
        num_objects = len(results)
        time_per_object = total_time / num_objects if num_objects else 0.0

        print(f"  Processing time: {total_time:.3f} s ({time_per_object:.3f} s per object)")

        # Save metadata (after timing calculated so we can include it)
        try:
            metadata_path = metadata_dir / f"{image_name}_metadata.csv"
            metadata_df = save_results_metadata(
                results,
                metadata_path,
                image_name,
                image_path,
                time_per_object_sec=time_per_object
            )
            if metadata_df is not None:
                print(f"  ✓ Saved metadata")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to save metadata: {e}")

        summary = {
            'image_path': image_path,
            'image_name': image_name,
            'num_points_total': total_points,
            'num_positive': num_positive,
            'num_negative': num_negative,
            'num_objects_segmented': len(results),
            'mean_score': np.mean(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'mean_iou': np.mean(ious) if ious else 0,
            'min_iou': np.min(ious) if ious else 0,
            'max_iou': np.max(ious) if ious else 0,
            'total_time_sec': total_time,
            'time_per_object_sec': time_per_object,
            'success': True,
            'error': None
        }

        print(f"  ✓ Success! {len(results)} positive objects segmented")
        if scores:
            print(f"    Mean score: {summary['mean_score']:.3f}, Range: [{summary['min_score']:.3f}, {summary['max_score']:.3f}]")
        if ious:
            print(f"    Mean IoU:   {summary['mean_iou']:.3f}, Range: [{summary['min_iou']:.3f}, {summary['max_iou']:.3f}]")

        return summary

    except Exception as e:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"  ✗ Failed to process image: {e}")
        import traceback
        print(f"  Error details: {traceback.format_exc()}")

        return {
            'image_path': image_path,
            'image_name': image_name,
            'num_points_total': total_points,
            'num_positive': num_positive,
            'num_negative': num_negative,
            'num_objects_segmented': 0,
            'mean_score': 0,
            'min_score': 0,
            'max_score': 0,
            'mean_iou': 0,
            'min_iou': 0,
            'max_iou': 0,
            'total_time_sec': total_time,
            'time_per_object_sec': 0.0,
            'success': False,
            'error': str(e)
        }


def process_dataset(pipeline, image_annotations, output_dir='results_evaluation',
                    max_images=None, visualize_individual=True, visualize_overview=True,
                    continue_on_failure=True):
    """
    Process entire dataset with positive/negative flags from `background` column.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    image_items = list(image_annotations.items())

    if max_images:
        image_items = image_items[:max_images]
        print(f"Processing limited to {max_images} images (set max_images=None for full dataset)")

    total_points = sum(len(data['points']) for _, data in image_items)
    total_positive = sum(sum(data.get('is_background', [])) for _, data in image_items)
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
                visualize_overview=visualize_overview
            )

            all_results.append(result)

            if result['success']:
                successful_images += 1
            else:
                failed_images += 1
                if not continue_on_failure:
                    print(f"Stopping processing due to failure")
                    break

        except Exception as e:
            failed_images += 1
            print(f"  ✗ Unexpected error: {e}")

            all_results.append({
                'image_path': image_path,
                'image_name': f"failed_image_{idx}",
                'num_points_total': len(data['points']),
                'num_positive': sum(data.get('is_background', [])),
                'num_negative': len(data['points']) - sum(data.get('is_background', [])),
                'num_objects_segmented': 0,
                'mean_score': 0,
                'min_score': 0,
                'max_score': 0,
                'mean_iou': 0,
                'min_iou': 0,
                'max_iou': 0,
                'total_time_sec': None,
                'time_per_object_sec': None,
                'success': False,
                'error': str(e)
            })

            if not continue_on_failure:
                break
            continue

    # Save summary CSV
    try:
        summary_df = pd.DataFrame(all_results)
        summary_path = Path(output_dir) / 'processing_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Saved processing summary to {summary_path}")
    except Exception as e:
        print(f"\n✗ Failed to save processing summary: {e}")

    successful_results = [r for r in all_results if r['success']]
    total_objects_segmented = sum(r['num_objects_segmented'] for r in all_results)

    print(f"\n{'=' * 70}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"  Total images processed: {len(all_results)}")
    print(f"  Successful: {successful_images}, Failed: {failed_images}")
    print(f"  Success rate: {successful_images / len(all_results) * 100:.1f}%")
    print(f"  Total positive objects segmented: {total_objects_segmented}")

    if successful_results:
        mean_scores = [r['mean_score'] for r in successful_results if r['num_objects_segmented'] > 0]
        mean_ious = [r['mean_iou'] for r in successful_results if r['num_objects_segmented'] > 0]
        if mean_scores:
            print(f"  Average confidence score (per image): {np.mean(mean_scores):.3f} ± {np.std(mean_scores):.3f}")
        if mean_ious:
            print(f"  Average IoU (per image):           {np.mean(mean_ious):.3f} ± {np.std(mean_ious):.3f}")

        times = [r['total_time_sec'] for r in successful_results if r.get('total_time_sec') is not None]
        times_per_obj = [r['time_per_object_sec'] for r in successful_results if r.get('time_per_object_sec') is not None]
        if times:
            print(f"  Average processing time (per image): {np.mean(times):.3f} s ± {np.std(times):.3f} s")
        if times_per_obj:
            print(f"  Average time per object:            {np.mean(times_per_obj):.3f} s ± {np.std(times_per_obj):.3f} s")

    print(f"\nResults saved to: {output_dir}/")
    if successful_images > 0:
        print(f"  - Masks: {output_dir}/masks/")
        print(f"  - Visualizations: {output_dir}/visualizations/")
        print(f"  - Metadata: {output_dir}/metadata/")

    return all_results


if __name__ == "__main__":
    start_time = time.time()
    print("Initializing SAM 2 pipeline with IoU evaluation...")

    try:
        pipeline = UnderwaterSAM2Pipeline(
            checkpoint_path='../sam2/checkpoints/sam2_hiera_small.pt',
            model_cfg='sam2_hiera_s.yaml',
            device='cpu'
        )
    except Exception as e:
        print(f"✗ Failed to initialize SAM 2 pipeline: {e}")
        exit(1)

    print("\nLoading annotations...")
    try:
        df = pd.read_csv(
            "../data/annotations-u45-Flinders_Western_Boundary_2011_targeted-Flinders_Western_Boundary_2011_targeted-copy-18164-35952eb26deb642336fe-dataframe.csv",
            low_memory=False
        )
        print(f"✓ Loaded {len(df)} annotations")

        # Check if background column exists
        if 'background' not in df.columns:
            print("⚠ Warning: 'background' column not found, treating all as positive")
            df['background'] = 1

        # Filter to images with Cups only
        if 'label_translated_lineage_names' not in df.columns:
            raise KeyError("Column 'label_translated_lineage_names' not found in CSV")

        df['label_translated_lineage_names'] = df['label_translated_lineage_names'].astype(str)

        mask_eck = df['label_translated_lineage_names'].str.contains(
            'Cup', case=False, na=False
        )

        eck_image_paths = df.loc[mask_eck, 'point_media_path_best'].unique()
        print(f"✓ Found {len(eck_image_paths)} images with Cups")

        df = df[df['point_media_path_best'].isin(eck_image_paths)].copy()
        print(f"✓ Filtered to {len(df)} annotations on Cups")

        # Background semantics: 1 = positive/use, 0 = negative/ignore
        num_positive = len(df[df['background'] == 1])
        num_negative = len(df[df['background'] == 0])
        print(f"  Positive points (background==1): {num_positive}")
        print(f"  Negative/ignored points (background==0): {num_negative}")

        print("\nFirst few filtered rows:")
        cols_to_show = [c for c in ['label_name', 'background', 'point_polygon'] if c in df.columns]
        print(df[cols_to_show].head(5).to_string())

    except Exception as e:
        print(f"✗ Failed to load annotations: {e}")
        print("Make sure your CSV file has these columns:")
        print("  - point_media_path_best (image URL or path)")
        print("  - point_x, point_y (normalized 0-1)")
        print("  - label_name (object label)")
        print("  - background (1=positive/use, 0=ignore)")
        print("  - label_translated_lineage_names (for filtering)")
        print("  - point_polygon (normalized polygon coordinates for IoU)")
        exit(1)

    print("\nPreparing annotations...")
    try:
        image_annotations = prepare_annotations_from_df(df)
        print(f"✓ Prepared annotations for {len(image_annotations)} unique images")

        total_objects = sum(len(data['points']) for data in image_annotations.values())
        print(f"✓ Total objects (rows) to process: {total_objects}")

    except Exception as e:
        print(f"✗ Failed to prepare annotations: {e}")
        exit(1)

    print("\nStarting dataset processing with IoU evaluation...")
    print("=" * 70)
    print("IMPORTANT: Processing 5 test images")
    print("Change max_images=None to process full dataset")
    print("Polygons are used ONLY for IoU, not as SAM2 prompts")
    print("=" * 70)

    try:
        results = process_dataset(
            pipeline=pipeline,
            image_annotations=image_annotations,
            output_dir='results_evaluation',
            max_images=None,
            visualize_individual=True,
            visualize_overview=True,
            continue_on_failure=True
        )

        print("\n✓ Pipeline complete!")
        print("\nNext steps:")
        print("  1. Check results_background_aware/visualizations/ for segmentation quality")
        print("  2. Check results_background_aware/metadata/ for per-object scores and IoU")
        print("  3. Inspect results_background_aware/processing_summary.csv for per-image metrics")
        print("  4. If quality is good, set max_images=None for full dataset")

        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user (Ctrl+C)")
        print("Partial results may be saved in results_background_aware/")

    except Exception as e:
        print(f"\n✗ Pipeline failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
