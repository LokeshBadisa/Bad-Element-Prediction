"""
JPEG-aware color variation detector.

Strategy:
  1. Convert to LAB (perceptual, decorrelates luma/chroma)
  2. Divide each channel into 8x8 DCT blocks (JPEG's native unit)
  3. Extract the DC coefficient (block mean) — this is the pre-compression
     signal; AC coefficients are mostly compression artifacts
  4. Compute variance of DC values across blocks
  5. Estimate the JPEG quantization noise floor and normalise against it
     so the returned score is dimensionless and threshold-stable
"""

import numpy as np
from PIL import Image


def _rgb_to_lab(img_array: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB array to CIE L*a*b* (float32)."""
    # Linearise sRGB
    rgb = img_array.astype(np.float32) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92

    # RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = rgb @ M.T

    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    # XYZ -> Lab
    epsilon, kappa = 0.008856, 903.3
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _block_dc_values(channel: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Return the DC coefficient (mean) of every non-overlapping block.
    Incomplete edge blocks are discarded so we only use clean 8x8 units.
    """
    h, w = channel.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    cropped = channel[:h_crop, :w_crop]
    # Reshape into (n_blocks_h, block_size, n_blocks_w, block_size)
    blocks = cropped.reshape(h_crop // block_size, block_size,
                             w_crop // block_size, block_size)
    # Mean over the two block axes → DC grid
    return blocks.mean(axis=(1, 3))   # shape: (n_h, n_w)


def _estimate_noise_floor(channel: np.ndarray, block_size: int = 8) -> float:
    """
    Estimate per-pixel JPEG quantization noise by measuring the
    mean absolute deviation *within* each block (intra-block AC energy).
    Under uniform compression this approximates sigma of the noise.
    Returns the expected std-dev of a block DC mean due to JPEG alone.
    """
    h, w = channel.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    cropped = channel[:h_crop, :w_crop]
    blocks = cropped.reshape(h_crop // block_size, block_size,
                             w_crop // block_size, block_size)
    block_means = blocks.mean(axis=(1, 3), keepdims=True)           # DC
    intra_mad = np.abs(blocks - block_means).mean(axis=(1, 3))      # AC proxy
    # DC noise = intra noise / sqrt(block_size^2)  (central limit)
    dc_noise_per_block = intra_mad.mean() / block_size
    return float(dc_noise_per_block)


def detect_color_variation(
    image: "str | np.ndarray | Image.Image",
    block_size: int = 8,
    min_blocks: int = 4,
) -> dict:
    """
    Detect whether a JPEG crop contains genuine color variation
    (i.e., variation that existed before compression).

    Parameters
    ----------
    image       : file path, PIL Image, or HxWx3 uint8 numpy array (RGB)
    block_size  : DCT block size — should stay 8 to match JPEG
    min_blocks  : minimum number of blocks required per axis for a
                  reliable estimate; raises ValueError if not met

    Returns
    -------
    dict with keys:
        score       – float ≥ 0; normalised variation score.
                      ~0  → uniform before compression
                      >1  → variation exceeds the JPEG noise floor
                      >>1 → clear genuine variation
        has_variation – bool  (score > 1.0)
        channel_scores – dict with per-channel (L, a, b) scores
        dc_variance   – raw variance of DC coefficients (L*a*b weighted)
        noise_floor   – estimated JPEG noise level (same units)
        n_blocks      – (rows, cols) of blocks analysed
    """
    # --- Load image ---------------------------------------------------------
    # if isinstance(image, str):
    #     img = np.array(Image.open(image).convert("RGB"))
    # elif isinstance(image, Image.Image):
    # img = np.array(image.convert("RGB"))
    # else:
    #     img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected an HxWx3 RGB image.")

    h, w = img.shape[:2]
    n_h = h // block_size
    n_w = w // block_size
    if n_h < min_blocks or n_w < min_blocks:
        raise ValueError(
            f"Crop too small: need at least {min_blocks}×{min_blocks} blocks "
            f"({min_blocks*block_size}px per side), got {h}×{w}px."
        )

    # --- Work in LAB --------------------------------------------------------
    lab = _rgb_to_lab(img)

    channel_names = ["L", "a", "b"]
    # Perceptual weights: chrominance channels (a,b) carry less signal
    # in typical scenes but are more diagnostic for colour variation
    channel_weights = [1.0, 1.5, 1.5]

    channel_scores = {}
    weighted_score_sum = 0.0
    weight_sum = sum(channel_weights)
    dc_variances = []
    noise_floors = []

    for i, (name, weight) in enumerate(zip(channel_names, channel_weights)):
        ch = lab[..., i]
        dc = _block_dc_values(ch, block_size)
        noise = _estimate_noise_floor(ch, block_size)

        # dc_var = float(dc.var())
        dc_std = float(dc.std())

        # Normalise: score = dc_std / noise  (both in same units)
        # Add small epsilon so we never divide by zero on synthetic images
        score = dc_std / (noise + 1e-6)
        # channel_scores[name] = round(score, 3)
        weighted_score_sum += weight * score
        # dc_variances.append(dc_var)
        # noise_floors.append(noise)

    overall_score = weighted_score_sum / weight_sum

    return {
        "score": round(overall_score, 3),
        "has_variation": overall_score > 1.0,
        # "channel_scores": channel_scores,
        # "dc_variance": round(float(np.mean(dc_variances)), 4),
        # "noise_floor": round(float(np.mean(noise_floors)), 4),
        # "n_blocks": (n_h, n_w),
    }

