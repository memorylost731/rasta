"""
File-based cache layer for pre-fetched city building data.

Checks for GeoJSON files on disk (written by prefetch-cities.py) before
falling back to a live Overpass query via buildings_lite.

Cache directory structure:
    data/cities/{city_id}/buildings.geojson

This module is intentionally zero-dependency beyond stdlib + buildings_lite.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Root of the pre-fetched city cache, relative to the repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _REPO_ROOT / "data" / "cities"

# ---------------------------------------------------------------------------
# Malta + Gozo city definitions (keep in sync with prefetch-cities.py)
# bbox order: [south, west, north, east]
# ---------------------------------------------------------------------------

PLANO_CITIES: List[Dict[str, Any]] = [
    # Malta
    {"id": "valletta",       "name": "Valletta",           "island": "Malta", "bbox": [35.893, 14.506, 35.905, 14.522]},
    {"id": "sliema",         "name": "Sliema",             "island": "Malta", "bbox": [35.907, 14.495, 35.920, 14.510]},
    {"id": "st-julians",     "name": "St Julian's",        "island": "Malta", "bbox": [35.912, 14.480, 35.930, 14.500]},
    {"id": "three-cities",   "name": "Three Cities",       "island": "Malta", "bbox": [35.880, 14.510, 35.895, 14.535]},
    {"id": "birkirkara",     "name": "Birkirkara",         "island": "Malta", "bbox": [35.888, 14.450, 35.905, 14.475]},
    {"id": "mosta",          "name": "Mosta",              "island": "Malta", "bbox": [35.900, 14.415, 35.920, 14.440]},
    {"id": "qormi",          "name": "Qormi",              "island": "Malta", "bbox": [35.868, 14.460, 35.885, 14.485]},
    {"id": "naxxar",         "name": "Naxxar",             "island": "Malta", "bbox": [35.905, 14.430, 35.925, 14.458]},
    {"id": "rabat-malta",    "name": "Rabat",              "island": "Malta", "bbox": [35.874, 14.388, 35.893, 14.412]},
    {"id": "mdina",          "name": "Mdina",              "island": "Malta", "bbox": [35.884, 14.399, 35.890, 14.407]},
    {"id": "marsaskala",     "name": "Marsaskala",         "island": "Malta", "bbox": [35.852, 14.555, 35.872, 14.578]},
    {"id": "mellieha",       "name": "Mellieha",           "island": "Malta", "bbox": [35.945, 14.348, 35.967, 14.378]},
    {"id": "swieqi",         "name": "Swieqi",             "island": "Malta", "bbox": [35.917, 14.472, 35.932, 14.490]},
    # Gozo
    {"id": "victoria-gozo",  "name": "Victoria (Rabat)",   "island": "Gozo",  "bbox": [36.036, 14.230, 36.052, 14.250]},
    {"id": "xlendi",         "name": "Xlendi",             "island": "Gozo",  "bbox": [36.023, 14.210, 36.033, 14.222]},
    {"id": "marsalforn",     "name": "Marsalforn",         "island": "Gozo",  "bbox": [36.063, 14.245, 36.077, 14.265]},
    {"id": "nadur",          "name": "Nadur",              "island": "Gozo",  "bbox": [36.028, 14.282, 36.046, 14.303]},
    {"id": "sannat",         "name": "Sannat",             "island": "Gozo",  "bbox": [36.014, 14.233, 36.030, 14.253]},
    {"id": "gharb",          "name": "Gharb",              "island": "Gozo",  "bbox": [36.050, 14.198, 36.068, 14.218]},
]


def get_cities() -> List[Dict[str, Any]]:
    """Return the full list of Malta + Gozo cities with their bboxes."""
    return list(PLANO_CITIES)


def find_city_by_id(city_id: str) -> Optional[Dict[str, Any]]:
    """Look up a city definition by its ID."""
    for city in PLANO_CITIES:
        if city["id"] == city_id:
            return city
    return None


def find_city_for_bbox(
    south: float, west: float, north: float, east: float,
    tolerance: float = 0.002,
) -> Optional[Dict[str, Any]]:
    """
    Find a pre-defined city whose bbox matches the requested area.

    Uses a tolerance of ~200m to account for minor differences in
    bbox parameters sent by the frontend vs the canonical definitions.
    """
    for city in PLANO_CITIES:
        cb = city["bbox"]
        if (
            abs(cb[0] - south) < tolerance
            and abs(cb[1] - west) < tolerance
            and abs(cb[2] - north) < tolerance
            and abs(cb[3] - east) < tolerance
        ):
            return city
    return None


def load_cached_buildings(city_id: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load pre-fetched buildings GeoJSON from disk for a given city.

    Returns the parsed GeoJSON dict, or None if the cache file
    does not exist or is unreadable.
    """
    root = cache_dir or CACHE_DIR
    path = root / city_id / "buildings.geojson"

    if not path.is_file():
        logger.debug("No cache file at %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data.get("features", []))
        logger.info("Loaded %d buildings from cache: %s", count, path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read cache %s: %s", path, exc)
        return None


def get_buildings(
    south: float, west: float, north: float, east: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Get building GeoJSON, checking file cache first, then live Overpass.

    Resolution order:
      1. Match requested bbox to a known city definition
      2. If matched, check disk cache at data/cities/{city_id}/buildings.geojson
      3. If cache hit, enrich with RF properties (prefetch files may lack them)
      4. If cache miss or no city match, query Overpass via buildings_lite

    This function never raises; on total failure it returns an empty
    FeatureCollection.
    """
    # Lazy import to avoid circular deps and keep this module importable
    # even if requests is somehow missing at import time
    from .buildings_lite import fetch_buildings_lite, _resolve_material_props, _extract_material

    # Step 1: try to match a known city
    city = find_city_for_bbox(south, west, north, east)

    if city is not None:
        # Step 2: try disk cache
        cached = load_cached_buildings(city["id"], cache_dir)

        if cached is not None:
            # Step 3: ensure RF properties exist on cached features
            # (prefetch-cities.py may have written files without rf_* fields)
            enriched = False
            for feat in cached.get("features", []):
                props = feat.get("properties", {})
                if "rf_attenuation_2_4ghz" not in props:
                    material = props.get("material") or _extract_material(props)
                    mat_props = _resolve_material_props(material)
                    props["material"] = material
                    props["color"] = mat_props["color"]
                    props["rf_attenuation_2_4ghz"] = mat_props["rf_db"]
                    props["rf_class"] = mat_props["rf_class"]
                    props["osm_id"] = str(props.get("osm_id", ""))
                    enriched = True

            if enriched:
                logger.info("Enriched cached features with RF properties")

            return cached

    # Step 4: live Overpass query
    logger.info("Cache miss — fetching live from Overpass")
    return fetch_buildings_lite(south, west, north, east)
