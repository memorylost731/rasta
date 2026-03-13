"""
PBR texture synthesis from material photographs.

Given a photo of a building surface, extracts a clean tileable texture patch
and generates Physically Based Rendering maps:
    - Diffuse / albedo map
    - Normal map (from Sobel gradient estimation)
    - Roughness map (from grayscale local variance)

Uses OpenCV for perspective correction, uniform region selection,
and Poisson-style seamless tiling at edges.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default tile output resolution
DEFAULT_TILE_SIZE = 512


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def _detect_and_correct_perspective(img: np.ndarray) -> np.ndarray:
    """
    Attempt perspective correction by finding the largest quadrilateral contour.
    If no clear quad is found, return the image as-is.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Find largest contour that approximates to a quadrilateral
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours_sorted[:10]:
        area = cv2.contourArea(cnt)
        img_area = img.shape[0] * img.shape[1]
        # Contour must cover at least 20% of image
        if area < img_area * 0.2:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = _order_points(pts)

            w = int(max(
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[3]),
            ))
            h = int(max(
                np.linalg.norm(pts[3] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
            ))

            if w < 100 or h < 100:
                continue

            dst = np.array([
                [0, 0], [w - 1, 0],
                [w - 1, h - 1], [0, h - 1],
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts, dst)
            corrected = cv2.warpPerspective(img, matrix, (w, h))
            logger.info("Perspective corrected to %dx%d", w, h)
            return corrected

    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ---------------------------------------------------------------------------
# Best uniform region selection
# ---------------------------------------------------------------------------

def _find_best_uniform_region(img: np.ndarray, tile_size: int) -> tuple[int, int]:
    """
    Find the most uniform (lowest variance) square region of the given size.
    Uses a sliding window with step = tile_size//4 for efficiency.
    Returns (x, y) of the top-left corner.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # If image is smaller than tile_size, return (0, 0)
    if h < tile_size or w < tile_size:
        return 0, 0

    step = max(tile_size // 4, 1)
    best_x, best_y = 0, 0
    best_score = float("inf")

    # Precompute integral images for fast variance calculation
    integral = cv2.integral(gray.astype(np.float64))
    integral_sq = cv2.integral(np.square(gray.astype(np.float64)))

    n = tile_size * tile_size

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            # Sum and sum-of-squares from integral images
            y2, x2 = y + tile_size, x + tile_size
            s = (integral[y2, x2] - integral[y, x2]
                 - integral[y2, x] + integral[y, x])
            s2 = (integral_sq[y2, x2] - integral_sq[y, x2]
                  - integral_sq[y2, x] + integral_sq[y, x])

            mean = s / n
            variance = (s2 / n) - (mean ** 2)

            # Score: lower variance = more uniform = better base texture
            # But not too uniform (might be a blank area) -- penalize very low variance
            if variance < 1.0:
                score = 1000.0  # penalize blank regions
            else:
                score = variance

            if score < best_score:
                best_score = score
                best_x, best_y = x, y

    return best_x, best_y


# ---------------------------------------------------------------------------
# Seamless tiling via edge blending
# ---------------------------------------------------------------------------

def _make_seamless_tile(patch: np.ndarray) -> np.ndarray:
    """
    Make a texture patch seamlessly tileable using mirror-blend at edges.

    Uses the approach of blending the patch with a half-offset version of itself
    to eliminate seams at tile boundaries.
    """
    h, w = patch.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Create offset version
    offset = np.roll(np.roll(patch, half_h, axis=0), half_w, axis=1)

    # Create diamond-shaped blending mask
    mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            # Distance from center, normalized
            dy = abs(y - half_h) / max(half_h, 1)
            dx = abs(x - half_w) / max(half_w, 1)
            mask[y, x] = max(0.0, min(1.0, (dy + dx)))

    # Smooth the mask to avoid hard transitions
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=h // 10, sigmaY=w // 10)

    if len(patch.shape) == 3:
        mask_3c = np.stack([mask] * patch.shape[2], axis=-1)
    else:
        mask_3c = mask

    blended = (patch.astype(np.float32) * (1.0 - mask_3c)
               + offset.astype(np.float32) * mask_3c)

    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PBR map generation
# ---------------------------------------------------------------------------

def _generate_normal_map(gray: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """
    Generate a tangent-space normal map from a grayscale heightfield
    using Sobel gradient estimation.

    Output: BGR image where:
        R = tangent (dX)   [128 = flat]
        G = bitangent (dY) [128 = flat]
        B = normal (Z)     [always ~255 = pointing up]
    """
    gray_f = gray.astype(np.float64) / 255.0

    # Compute gradients
    sobel_x = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)

    # Scale gradients by strength
    sobel_x *= strength
    sobel_y *= strength

    # Construct normal vectors (tangent space)
    # N = normalize(-dX, -dY, 1.0)
    nx = -sobel_x
    ny = -sobel_y
    nz = np.ones_like(nx)

    # Normalize
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length = np.maximum(length, 1e-8)
    nx /= length
    ny /= length
    nz /= length

    # Map from [-1, 1] to [0, 255]
    r = ((nx + 1.0) * 0.5 * 255.0).astype(np.uint8)  # tangent
    g = ((ny + 1.0) * 0.5 * 255.0).astype(np.uint8)  # bitangent
    b = ((nz + 1.0) * 0.5 * 255.0).astype(np.uint8)  # normal (mostly 255)

    normal_map = cv2.merge([b, g, r])  # BGR order for OpenCV
    return normal_map


def _generate_roughness_map(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Generate a roughness map from local grayscale variance.

    High variance regions = rough surface = white (255)
    Low variance regions  = smooth surface = dark (0)
    """
    gray_f = gray.astype(np.float64)

    # Local mean
    mean = cv2.blur(gray_f, (kernel_size, kernel_size))

    # Local mean of squares
    mean_sq = cv2.blur(gray_f ** 2, (kernel_size, kernel_size))

    # Local variance
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 0)

    # Normalize to 0-255
    std_dev = np.sqrt(variance)
    max_std = std_dev.max()
    if max_std > 0:
        roughness = (std_dev / max_std * 255.0).astype(np.uint8)
    else:
        roughness = np.full_like(gray, 128, dtype=np.uint8)

    # Apply slight blur for smoother map
    roughness = cv2.GaussianBlur(roughness, (5, 5), 0)

    return roughness


# ---------------------------------------------------------------------------
# Tile size estimation
# ---------------------------------------------------------------------------

# Approximate real-world tile sizes by material (cm per texture tile)
_TILE_SIZE_CM: dict[str, float] = {
    "concrete": 100.0,
    "brick": 25.0,
    "marble": 60.0,
    "granite": 60.0,
    "limestone": 60.0,
    "wood_plank": 120.0,
    "wood_panel": 244.0,
    "ceramic_tile": 30.0,
    "porcelain_tile": 60.0,
    "glass": 100.0,
    "plaster": 100.0,
    "stucco": 100.0,
    "metal_sheet": 200.0,
    "metal_panel": 120.0,
    "stone": 40.0,
    "terrazzo": 60.0,
    "vinyl": 30.0,
    "carpet": 50.0,
    "linoleum": 200.0,
    "cork": 30.0,
    "slate": 30.0,
    "sandstone": 40.0,
    "render": 100.0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_texture(
    image_path: str,
    material: str,
    output_dir: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    normal_strength: float = 2.0,
    roughness_kernel: int = 9,
) -> dict:
    """
    Extract a clean tileable texture from a photograph and generate PBR maps.

    Pipeline:
        1. Load image
        2. Perspective correction (if a quad surface is detected)
        3. Select the most uniform region
        4. Crop to tile_size square
        5. Make seamless tileable (mirror-blend)
        6. Generate normal map (Sobel gradients)
        7. Generate roughness map (local variance)
        8. Save all maps to output_dir

    Args:
        image_path:       Path to source photograph.
        material:         Material name (used for output filenames and tile size lookup).
        output_dir:       Directory to save generated textures.
        tile_size:        Output texture resolution in pixels (default 512).
        normal_strength:  Normal map gradient multiplier (default 2.0).
        roughness_kernel: Kernel size for roughness variance calculation (default 9).

    Returns:
        {
            "diffuse":       str,   # path to diffuse/albedo map
            "normal":        str,   # path to normal map
            "roughness":     str,   # path to roughness map
            "tile_size_cm":  float, # estimated real-world size of one tile
        }

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If image cannot be loaded or is too small.
    """
    src_path = Path(image_path)
    out_path = Path(output_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Load
    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]
    logger.info("Source image: %dx%d", w, h)

    # Step 1: Perspective correction
    corrected = _detect_and_correct_perspective(img)

    # Step 2: Ensure we have enough resolution
    ch, cw = corrected.shape[:2]
    effective_tile = min(tile_size, min(ch, cw))
    if effective_tile < 64:
        raise ValueError(
            f"Image too small after correction ({cw}x{ch}). "
            f"Need at least 64x64 for texture extraction."
        )

    # Step 3: Find most uniform region
    bx, by = _find_best_uniform_region(corrected, effective_tile)

    # Step 4: Crop
    patch = corrected[by:by + effective_tile, bx:bx + effective_tile]

    # Resize to target tile_size if patch is smaller
    if effective_tile != tile_size:
        patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LANCZOS4)

    # Step 5: Make seamless
    seamless = _make_seamless_tile(patch)

    # Step 6: Generate PBR maps
    gray = cv2.cvtColor(seamless, cv2.COLOR_BGR2GRAY)
    normal_map = _generate_normal_map(gray, strength=normal_strength)
    roughness_map = _generate_roughness_map(gray, kernel_size=roughness_kernel)

    # Step 7: Save
    safe_name = material.replace(" ", "_").lower()
    diffuse_path = out_path / f"{safe_name}_diffuse.jpg"
    normal_path = out_path / f"{safe_name}_normal.png"
    roughness_path = out_path / f"{safe_name}_roughness.png"

    cv2.imwrite(str(diffuse_path), seamless, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(normal_path), normal_map)
    cv2.imwrite(str(roughness_path), roughness_map)

    tile_cm = _TILE_SIZE_CM.get(material, 50.0)

    logger.info(
        "Texture extracted: diffuse=%s normal=%s roughness=%s tile=%.0fcm",
        diffuse_path, normal_path, roughness_path, tile_cm,
    )

    return {
        "diffuse": str(diffuse_path),
        "normal": str(normal_path),
        "roughness": str(roughness_path),
        "tile_size_cm": tile_cm,
    }
