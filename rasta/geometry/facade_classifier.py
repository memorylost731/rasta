"""
Facade material classification from street-view imagery.

Takes a building footprint centroid + nearby Mapillary images, crops
facade regions, runs Rasta's texture_identify on each crop, and
aggregates per-building material classifications with confidence.

Falls back to OSM material tags when no imagery is available.
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependency on cv2/numpy at import time
_cv2 = None
_np = None


def _ensure_cv2():
    global _cv2, _np
    if _cv2 is None:
        try:
            import cv2
            import numpy as np
            _cv2 = cv2
            _np = np
        except ImportError:
            raise ImportError(
                "opencv-python-headless is required for facade classification. "
                "Install with: pip install opencv-python-headless"
            )


# Import Rasta material DB and classifier
try:
    from rasta.texture_identify import identify_material, MATERIAL_DB, ALL_MATERIALS
except ImportError:
    # Allow standalone testing
    MATERIAL_DB = {}
    ALL_MATERIALS = []

    def identify_material(image_path: str) -> dict:
        return {"material": "limestone", "confidence": 0.5, "method": "stub"}


# ---------------------------------------------------------------------------
# Facade region of interest extraction
# ---------------------------------------------------------------------------

@dataclass
class FacadeCrop:
    """A cropped facade region from a street-view image."""
    image_path: Path
    source_image_id: str
    crop_path: Optional[Path] = None
    classification: Optional[dict] = None


@dataclass
class BuildingClassification:
    """Aggregated facade classification for a single building."""
    osm_id: int
    material: str
    confidence: float
    method: str  # "mapillary", "osm_tag", "default"
    subcategory: str = ""
    crop_count: int = 0
    classifications: list[dict] = field(default_factory=list)
    rf_attenuation_db: Optional[dict] = None


# OSM material tag to Rasta material name mapping
_OSM_TO_RASTA: dict[str, str] = {
    "limestone": "limestone",
    "sandstone": "sandstone",
    "stone": "stone",
    "concrete": "concrete",
    "brick": "brick",
    "glass": "glass",
    "metal": "metal_sheet",
    "steel": "metal_sheet",
    "aluminium": "metal_panel",
    "aluminum": "metal_panel",
    "wood": "wood_plank",
    "timber": "wood_plank",
    "plaster": "plaster",
    "stucco": "stucco",
    "render": "render",
    "marble": "marble",
    "granite": "granite",
    "slate": "slate",
    "ceramic": "ceramic_tile",
    "cement_block": "concrete",
    "mud": "render",
}

# Valletta default: most buildings are globigerina limestone
VALLETTA_DEFAULT_MATERIAL = "limestone"


def _map_osm_material(osm_material: Optional[str]) -> Optional[str]:
    """Map an OSM material tag value to a Rasta material name."""
    if not osm_material:
        return None
    normalized = osm_material.lower().strip().replace("-", "_").replace(" ", "_")
    if normalized in _OSM_TO_RASTA:
        return _OSM_TO_RASTA[normalized]
    # Substring match
    for key, rasta_name in _OSM_TO_RASTA.items():
        if key in normalized:
            return rasta_name
    return None


def _crop_facade_region(
    image_path: Path,
    output_path: Path,
    vertical_crop: tuple[float, float] = (0.15, 0.85),
    horizontal_crop: tuple[float, float] = (0.1, 0.9),
) -> Optional[Path]:
    """
    Crop the facade region from a street-view image.

    Street-view images typically have sky at top and ground/cars at bottom.
    We crop the middle band which is most likely to contain facade material.

    Args:
        image_path: Source image path.
        output_path: Where to save the crop.
        vertical_crop: (top_ratio, bottom_ratio) of image height.
        horizontal_crop: (left_ratio, right_ratio) of image width.

    Returns:
        Path to cropped image, or None on failure.
    """
    _ensure_cv2()

    img = _cv2.imread(str(image_path))
    if img is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    h, w = img.shape[:2]
    y1 = int(h * vertical_crop[0])
    y2 = int(h * vertical_crop[1])
    x1 = int(w * horizontal_crop[0])
    x2 = int(w * horizontal_crop[1])

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("Empty crop from %s", image_path)
        return None

    _cv2.imwrite(str(output_path), crop)
    return output_path


def classify_facade(
    image_path: str,
    source_image_id: str = "",
) -> dict:
    """
    Classify facade material from a single street-view image.

    Crops the facade region and runs Rasta's texture_identify pipeline
    (Ollama vision first, OpenCV fallback).

    Args:
        image_path: Path to the street-view photograph.
        source_image_id: Mapillary image ID for provenance tracking.

    Returns:
        Classification dict with material, confidence, method, and RF properties.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Crop facade region to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp") as tmp:
        crop_path = Path(tmp.name)

    cropped = _crop_facade_region(path, crop_path)
    try:
        if cropped is None:
            # Fall back to full image
            result = identify_material(str(path))
        else:
            result = identify_material(str(cropped))
    finally:
        try:
            crop_path.unlink(missing_ok=True)
        except OSError:
            pass

    result["source_image_id"] = source_image_id
    result["facade_crop"] = cropped is not None
    return result


def classify_building(
    osm_id: int,
    osm_material: Optional[str],
    image_paths: Optional[list[tuple[str, str]]] = None,
) -> BuildingClassification:
    """
    Classify a building's facade material from available data.

    Priority:
        1. Mapillary street-view images (highest confidence)
        2. OSM material tags (medium confidence)
        3. Valletta default limestone (low confidence)

    Args:
        osm_id: OpenStreetMap building ID.
        osm_material: Material tag from OSM (may be None).
        image_paths: List of (file_path, mapillary_image_id) tuples.

    Returns:
        BuildingClassification with aggregated result.
    """
    classifications: list[dict] = []

    # Try Mapillary images
    if image_paths:
        for fpath, img_id in image_paths:
            try:
                result = classify_facade(fpath, source_image_id=img_id)
                classifications.append(result)
            except Exception as exc:
                logger.warning("Facade classification failed for %s: %s", fpath, exc)

    # Aggregate image-based classifications
    if classifications:
        material, confidence, subcategory = _aggregate_classifications(classifications)
        method = "mapillary"
    elif osm_material:
        rasta_material = _map_osm_material(osm_material)
        if rasta_material and rasta_material in MATERIAL_DB:
            material = rasta_material
            confidence = 0.6
            subcategory = MATERIAL_DB[material]["subcategory"]
            method = "osm_tag"
        else:
            material = VALLETTA_DEFAULT_MATERIAL
            confidence = 0.3
            subcategory = MATERIAL_DB.get(material, {}).get("subcategory", "")
            method = "default"
    else:
        material = VALLETTA_DEFAULT_MATERIAL
        confidence = 0.3
        subcategory = MATERIAL_DB.get(material, {}).get("subcategory", "natural")
        method = "default"

    # Get RF properties
    rf = None
    if material in MATERIAL_DB:
        rf = MATERIAL_DB[material]["rf_attenuation_db"]

    return BuildingClassification(
        osm_id=osm_id,
        material=material,
        confidence=round(confidence, 3),
        method=method,
        subcategory=subcategory,
        crop_count=len(classifications),
        classifications=classifications,
        rf_attenuation_db=rf,
    )


def _aggregate_classifications(classifications: list[dict]) -> tuple[str, float, str]:
    """
    Aggregate multiple facade classifications into a single result.

    Uses weighted voting by confidence score.

    Returns:
        (material, confidence, subcategory)
    """
    votes: dict[str, float] = {}
    subcategories: dict[str, str] = {}

    for cls in classifications:
        mat = cls.get("material", "")
        conf = cls.get("confidence", 0.5)
        votes[mat] = votes.get(mat, 0) + conf
        if mat not in subcategories:
            subcategories[mat] = cls.get("subcategory", "")

    if not votes:
        return VALLETTA_DEFAULT_MATERIAL, 0.3, ""

    # Winner takes all, confidence = average of winning material's scores
    winner = max(votes, key=lambda k: votes[k])
    winner_scores = [c["confidence"] for c in classifications if c.get("material") == winner]
    avg_confidence = sum(winner_scores) / len(winner_scores) if winner_scores else 0.5

    # Boost confidence when multiple images agree
    agreement_ratio = len(winner_scores) / len(classifications)
    boosted = avg_confidence * (0.8 + 0.2 * agreement_ratio)
    final_confidence = min(boosted, 0.95)

    return winner, final_confidence, subcategories.get(winner, "")
