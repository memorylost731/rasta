"""
Zero-dependency building footprint fetcher for the PlanO pipeline.

Uses only `requests` + stdlib to query the Overpass API and convert
results to a GeoJSON FeatureCollection enriched with:
    - 3D extrusion properties (height, min_height, levels)
    - Material classification with Malta/Gozo defaults (limestone)
    - RF attenuation at 2.4 GHz per material
    - RF penetration class (transparent/light/medium/heavy/opaque)
    - Rendering color (hex) derived from material

Designed to run on the GPU server where shapely, overpy, and httpx
are NOT available.  Only stdlib + requests.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 120  # seconds for the Overpass query itself
HTTP_TIMEOUT = 150  # seconds for the HTTP request (> Overpass timeout)

METERS_PER_LEVEL = 3.2
DEFAULT_HEIGHT_M = 10.0
MIN_HEIGHT_M = 3.0

# Valletta default bounding box (south, west, north, east)
VALLETTA_BBOX = (35.893, 14.506, 35.905, 14.522)

# ---------------------------------------------------------------------------
# Material properties lookup (Malta / Mediterranean context)
# ---------------------------------------------------------------------------

MATERIAL_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "limestone": {"color": "#f5e6c8", "rf_db": 10.0, "rf_class": "medium", "thermal": 1.3},
    "concrete":  {"color": "#b0b0b0", "rf_db": 15.0, "rf_class": "heavy",  "thermal": 1.7},
    "brick":     {"color": "#c4735b", "rf_db": 8.0,  "rf_class": "medium", "thermal": 0.8},
    "glass":     {"color": "#c8e0f0", "rf_db": 2.5,  "rf_class": "light",  "thermal": 1.0},
    "stone":     {"color": "#d4c5a9", "rf_db": 12.0, "rf_class": "heavy",  "thermal": 2.3},
    "metal":     {"color": "#8a8a8a", "rf_db": 30.0, "rf_class": "opaque", "thermal": 50.0},
    "wood":      {"color": "#c49a6c", "rf_db": 3.0,  "rf_class": "light",  "thermal": 0.15},
    "plaster":   {"color": "#f0ead6", "rf_db": 4.0,  "rf_class": "light",  "thermal": 0.5},
    "render":    {"color": "#e8dcc8", "rf_db": 4.0,  "rf_class": "light",  "thermal": 0.5},
    "marble":    {"color": "#f0f0f0", "rf_db": 14.0, "rf_class": "heavy",  "thermal": 2.8},
    "sandstone": {"color": "#e6d5a8", "rf_db": 9.0,  "rf_class": "medium", "thermal": 1.7},
}

DEFAULT_MATERIAL = "limestone"

# ---------------------------------------------------------------------------
# In-memory cache with TTL
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 3600  # 1 hour


def _cache_key(south: float, west: float, north: float, east: float) -> str:
    return f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if key in _CACHE:
        ts, data = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            logger.debug("Cache hit for %s", key)
            return data
        del _CACHE[key]
    return None


def _cache_put(key: str, data: Dict[str, Any]) -> None:
    # Evict expired entries when cache grows beyond 100
    if len(_CACHE) > 100:
        now = time.time()
        expired = [k for k, (ts, _) in _CACHE.items() if now - ts >= CACHE_TTL]
        for k in expired:
            del _CACHE[k]
    _CACHE[key] = (time.time(), data)


# ---------------------------------------------------------------------------
# Overpass query + HTTP
# ---------------------------------------------------------------------------

def _build_query(south: float, west: float, north: float, east: float) -> str:
    bb = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["building"]({bb});
  relation["building"]({bb});
);
out body;
>;
out skel qt;
"""


def _overpass_fetch(query: str, retries: int = 3, backoff: float = 10.0) -> Dict[str, Any]:
    """
    POST query to Overpass API with retry on 429/503/504.

    Returns parsed JSON dict.  On persistent failure returns {"elements": []}.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"Accept": "application/json"},
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 503, 504):
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Overpass HTTP %d — retry %d/%d in %.0fs",
                    resp.status_code, attempt, retries, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()

        except requests.exceptions.Timeout:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Overpass timeout — retry %d/%d in %.0fs", attempt, retries, wait)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Overpass connection error: %s — retry in %.0fs", exc, wait)
            time.sleep(wait)

    logger.error("Overpass query failed after %d attempts", retries)
    return {"elements": []}


# ---------------------------------------------------------------------------
# OSM tag parsing
# ---------------------------------------------------------------------------

def _parse_height(tags: Dict[str, str]) -> float:
    """Extract or estimate building height in meters."""
    raw = tags.get("height", "")
    if raw:
        try:
            cleaned = raw.strip().lower()
            if cleaned.endswith("ft"):
                return float(cleaned.replace("ft", "").strip()) * 0.3048
            return float(cleaned.replace("m", "").strip())
        except (ValueError, TypeError):
            pass

    levels_str = tags.get("building:levels", tags.get("levels", ""))
    if levels_str:
        try:
            levels = int(float(levels_str))
            return max(levels * METERS_PER_LEVEL, MIN_HEIGHT_M)
        except (ValueError, TypeError):
            pass

    return DEFAULT_HEIGHT_M


def _parse_min_height(tags: Dict[str, str]) -> float:
    raw = tags.get("min_height", "0")
    try:
        return float(raw.replace("m", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> Optional[int]:
    if not val:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _extract_material(tags: Dict[str, str]) -> str:
    """Extract facade/building material, defaulting to limestone for Malta."""
    for key in (
        "building:facade:material",
        "building:material",
        "building:cladding",
        "facade:material",
        "material",
        "surface",
    ):
        val = tags.get(key)
        if val:
            return val.lower().strip()
    return DEFAULT_MATERIAL


def _resolve_material_props(material: str) -> Dict[str, Any]:
    """Look up material properties.  Falls back through substring match, then default."""
    # Exact match first
    if material in MATERIAL_PROPERTIES:
        return MATERIAL_PROPERTIES[material]
    # Substring match (e.g. "reinforced_concrete" -> "concrete")
    for key, props in MATERIAL_PROPERTIES.items():
        if key in material:
            return props
    return MATERIAL_PROPERTIES[DEFAULT_MATERIAL]


# ---------------------------------------------------------------------------
# Overpass JSON -> GeoJSON conversion
# ---------------------------------------------------------------------------

def _nodes_to_coords(
    node_ids: List[int],
    node_map: Dict[int, Tuple[float, float]],
) -> List[List[float]]:
    """Convert node IDs to [lon, lat] coordinate ring (GeoJSON winding)."""
    coords = []
    for nid in node_ids:
        if nid in node_map:
            lat, lon = node_map[nid]
            coords.append([lon, lat])
    # Close the ring
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _build_feature(
    osm_id: int,
    osm_type: str,
    tags: Dict[str, str],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single GeoJSON Feature with all enriched properties."""
    height = _parse_height(tags)
    min_height = _parse_min_height(tags)
    levels = _safe_int(tags.get("building:levels", ""))
    material = _extract_material(tags)
    mat_props = _resolve_material_props(material)

    return {
        "type": "Feature",
        "id": osm_id,
        "geometry": geometry,
        "properties": {
            "osm_id": str(osm_id),
            "osm_type": osm_type,
            "building": tags.get("building", "yes"),
            "name": tags.get("name", ""),
            "addr:street": tags.get("addr:street", ""),
            "addr:housenumber": tags.get("addr:housenumber", ""),
            "height": round(height, 1),
            "min_height": round(min_height, 1),
            "levels": levels,
            "material": material,
            "color": mat_props["color"],
            "rf_attenuation_2_4ghz": mat_props["rf_db"],
            "rf_class": mat_props["rf_class"],
        },
    }


def _parse_overpass_to_geojson(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw Overpass JSON to a GeoJSON FeatureCollection.

    Handles:
      - Standalone ways (simple polygons)
      - Relations with outer/inner members (multipolygons with holes)
      - Node resolution (way node IDs -> lat/lon coordinates)
    """
    elements = data.get("elements", [])

    # Pass 1: build lookup tables
    node_map: Dict[int, Tuple[float, float]] = {}
    way_nodes: Dict[int, List[int]] = {}
    ways: List[Dict] = []
    relations: List[Dict] = []

    for elem in elements:
        etype = elem.get("type")
        if etype == "node":
            node_map[elem["id"]] = (elem["lat"], elem["lon"])
        elif etype == "way":
            if "nodes" in elem:
                way_nodes[elem["id"]] = elem["nodes"]
            if "tags" in elem and "building" in elem.get("tags", {}):
                ways.append(elem)
        elif etype == "relation":
            if "tags" in elem and "building" in elem.get("tags", {}):
                relations.append(elem)

    features: List[Dict[str, Any]] = []

    # Pass 2: standalone building ways -> Polygon features
    for w in ways:
        nids = w.get("nodes", [])
        coords = _nodes_to_coords(nids, node_map)
        if len(coords) < 4:
            continue

        geometry = {"type": "Polygon", "coordinates": [coords]}
        features.append(_build_feature(w["id"], "way", w.get("tags", {}), geometry))

    # Pass 3: multipolygon relations -> Polygon features
    for rel in relations:
        tags = rel.get("tags", {})
        members = rel.get("members", [])

        outer_rings: List[List[List[float]]] = []
        inner_rings: List[List[List[float]]] = []

        for member in members:
            if member.get("type") != "way":
                continue
            role = member.get("role", "outer")
            nids = way_nodes.get(member["ref"], [])
            coords = _nodes_to_coords(nids, node_map)
            if len(coords) < 4:
                continue
            if role == "inner":
                inner_rings.append(coords)
            else:
                outer_rings.append(coords)

        if not outer_rings:
            continue

        # Primary polygon: first outer ring + all inner rings (holes)
        polygon_coords = [outer_rings[0]] + inner_rings
        geometry = {"type": "Polygon", "coordinates": polygon_coords}
        features.append(_build_feature(rel["id"], "relation", tags, geometry))

        # Extra outer rings as separate features (rare)
        for extra in outer_rings[1:]:
            geometry = {"type": "Polygon", "coordinates": [extra]}
            features.append(_build_feature(rel["id"], "relation", tags, geometry))

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "count": len(features),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "default_material": DEFAULT_MATERIAL,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_buildings_lite(
    south: float = VALLETTA_BBOX[0],
    west: float = VALLETTA_BBOX[1],
    north: float = VALLETTA_BBOX[2],
    east: float = VALLETTA_BBOX[3],
) -> Dict[str, Any]:
    """
    Fetch enriched building GeoJSON for a bounding box.

    Returns a GeoJSON FeatureCollection with per-building properties
    including height, material, color, RF attenuation, and RF class.

    Results are cached in memory for 1 hour keyed by bbox.
    Falls back to empty FeatureCollection on Overpass failure.
    """
    key = _cache_key(south, west, north, east)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    logger.info("Fetching buildings for bbox [%.5f,%.5f,%.5f,%.5f]", south, west, north, east)
    query = _build_query(south, west, north, east)
    raw = _overpass_fetch(query)
    geojson = _parse_overpass_to_geojson(raw)

    _cache_put(key, geojson)
    logger.info("Fetched %d buildings", geojson["metadata"]["count"])
    return geojson


def get_material_properties() -> Dict[str, Dict[str, Any]]:
    """Return the full material properties lookup table."""
    return dict(MATERIAL_PROPERTIES)
