"""
BBox Variation IoU Analyzer
----------------------------
1. Compute union bbox from N input annotations
2. Crop that region from the JPEG image
3. Suppress JPEG artifacts with Gaussian blur
4. Detect variation via Sobel edge magnitude
5. Find the largest bbox enclosing all variation pixels
6. Compute IoU between that variation bbox and each input bbox
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


# ─── Types ────────────────────────────────────────────────────────────────────
BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)


# ─── Step 1: Union BBox ───────────────────────────────────────────────────────
def union_bbox(bboxes: List[BBox]) -> BBox:
    """Smallest bbox containing all input bboxes."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (int(x1), int(y1), int(x2), int(y2))


# ─── Step 2: Crop image to bbox ───────────────────────────────────────────────
def crop_to_bbox(img: np.ndarray, bbox: BBox) -> Tuple[np.ndarray, BBox]:
    """
    Crop image to bbox, clamped to image boundaries.
    Returns the crop and the clamped bbox (in image coords).
    """
    h, w = img.shape[:2]
    x1 = max(0, bbox[0]);  y1 = max(0, bbox[1])
    x2 = min(w, bbox[2]);  y2 = min(h, bbox[3])
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ─── Step 3: JPEG artifact suppression ───────────────────────────────────────
def suppress_jpeg_artifacts(crop: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian blur to attenuate 8×8 DCT block-boundary artifacts in JPEG images.
    Kernel size is automatically derived from sigma (odd, >= 3).
    """
    if blur_sigma <= 0:
        return crop
    ksize = int(2 * np.ceil(2 * blur_sigma) + 1)   # ~4σ wide, always odd
    try:
        return cv2.GaussianBlur(crop, (ksize, ksize), blur_sigma)
    except:
        return crop


# ─── Step 4: Variation detection (Sobel) ─────────────────────────────────────
def compute_variation_map(crop: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel variation as Sobel edge magnitude on grayscale crop.
    Returns float32 array in [0, ~1450] range (unnormalized).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude


# ─── Step 5: Variation bbox ───────────────────────────────────────────────────
def variation_bbox(
    magnitude: np.ndarray,
    crop_origin: Tuple[int, int],
    threshold: float = 20.0,
    min_area_ratio: float = 0.01,
) -> Optional[BBox]:
    """
    Threshold the magnitude map and find the tightest bbox
    enclosing all variation pixels.

    Args:
        magnitude:      Sobel magnitude map of the crop (H×W float32)
        crop_origin:    (x_offset, y_offset) of the crop in original image coords
        threshold:      Minimum magnitude to count as variation
        min_area_ratio: Ignore result if variation pixels < this fraction of crop area

    Returns:
        BBox in original image coordinates, or None if no variation found.
    """
    mask = magnitude >= threshold
    n_var = mask.sum()
    total = magnitude.size

    if n_var < max(4, int(total * min_area_ratio)):
        return None

    ys, xs = np.where(mask)
    ox, oy = crop_origin
    x1, y1 = int(xs.min()) + ox, int(ys.min()) + oy
    x2, y2 = int(xs.max()) + ox, int(ys.max()) + oy
    return (x1, y1, x2, y2)


# ─── Step 6: IoU ─────────────────────────────────────────────────────────────
def compute_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union between two bboxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ─── Full Pipeline ────────────────────────────────────────────────────────────
def get_IoU_list(
    img,
    bboxes: List[BBox],
    blur_sigma: float = 1.0,
    var_threshold: float = 20.0,
    min_area_ratio: float = 0.01,
    visualize: bool = False,
    save_vis: Optional[str] = None,
) -> dict:
    """
    Full pipeline.

    Args:
        image_path:     Path to the JPEG image.
        bboxes:         List of (x1, y1, x2, y2) annotations.
        blur_sigma:     Gaussian blur sigma for JPEG artifact suppression (0 = off).
        var_threshold:  Sobel magnitude threshold for variation detection.
        min_area_ratio: Minimum fraction of crop area that must be variation.
        visualize:      If True, show result with cv2.imshow.
        save_vis:       If set, save visualisation to this path.

    Returns:
        dict with keys:
            union_bbox      – BBox
            variation_bbox  – BBox or None
            ious            – List[float], one per input bbox
            magnitude_map   – np.ndarray (the Sobel map of the crop)
    """
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1. Union bbox
    u_bbox = union_bbox(bboxes)
    # print(f"[1] Union BBox       : {u_bbox}")

    # 2. Crop

    crop, clamped = crop_to_bbox(img, u_bbox)
    if crop.size == 0:
        return {
            "union_bbox":     u_bbox,
            "variation_bbox": None,
            "ious":           [0.0] * len(bboxes),
            "magnitude_map":  None,
        }
    crop_size = crop.size
    # print(f"[2] Crop (clamped)   : {clamped}  size={crop.shape[1]}×{crop.shape[0]}")

    # 3. JPEG artifact suppression
    blurred = suppress_jpeg_artifacts(crop, blur_sigma)
    # print(f"[3] Blur sigma       : {blur_sigma}")

    # 4. Variation map

    mag = compute_variation_map(blurred)
    
    # print(f"[4] Magnitude range  : [{mag.min():.1f}, {mag.max():.1f}]")

    # 5. Variation bbox
    ox, oy = clamped[0], clamped[1]
    v_bbox = variation_bbox(mag, (ox, oy), var_threshold, min_area_ratio)
    # if v_bbox is None:
    #     print(f"[5] Variation BBox   : None (no variation above threshold)")
    # else:
    #     print(f"[5] Variation BBox   : {v_bbox}")

    # 6. IoU per bbox
    ious = []
    # print(f"\n{'─'*52}")
    # print(f"{'#':<5} {'BBox':<24} {'IoU':>8}  Category")
    # print(f"{'─'*52}")
    for i, b in enumerate(bboxes):
        v = compute_iou(v_bbox, b) if v_bbox else 0.0
        ious.append(v)
        cat = "HIGH" if v >= 0.7 else "MED" if v >= 0.4 else "LOW" if v > 0 else "NONE"
        # print(f"#{i+1:<4} {str(b):<24} {v:>8.4f}  {cat}")
    # print(f"{'─'*52}")
    # print(f"{'Mean':>30} {np.mean(ious):>8.4f}")
    # print(f"{'Max':>30}  {np.max(ious):>8.4f}")
    # print(f"{'Min':>30}  {np.min(ious):>8.4f}")

    # Visualization
    if visualize or save_vis:
        _draw_result(img, bboxes, u_bbox, v_bbox, ious, mag, clamped, visualize, save_vis)

    return {
        "union_bbox":      u_bbox,
        "variation_bbox":  v_bbox,
        "ious":            ious,
        "magnitude_map":   mag,
    }

def group_intersecting_bboxes(bbox_dict):
    """
    Group bounding boxes that intersect with each other.
    
    Args:
        bbox_dict: Dict where key is box number/id and value is [x1, y1, x2, y2]
    
    Returns:
        List of groups, where each group is a dict of {box_id: coords} that intersect
    """
    def intersects(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def isInside(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    keys = list(bbox_dict.keys())
    n = len(keys)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if intersects(bbox_dict[keys[i]], bbox_dict[keys[j]]) and\
                not isInside(bbox_dict[keys[i]], bbox_dict[keys[j]]) and\
                not isInside(bbox_dict[keys[j]], bbox_dict[keys[i]]) and\
                compute_iou(bbox_dict[keys[i]], bbox_dict[keys[j]]) > 0.2:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = {}
        groups[root][keys[i]] = bbox_dict[keys[i]]

    return list(groups.values())