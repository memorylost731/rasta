"""
OSM Overpass API client for building footprint extraction.

Fetches building polygons with height, levels, material, and roof tags,
converts them to GeoJSON FeatureCollections ready for MapLibre fill-extrusion.

Default area: Valletta, Malta (35.8989, 14.5146).
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 60

# Valletta, Malta bounding box (south, west, north, east)
VALLETTA_BBOX = (35.8940, 14.5080, 35.9040, 14.5200)

# Height estimation defaults
METERS_PER_LEVEL = 3.2
DEFAULT_HEIGHT_M = 9.6  # 3 levels fallback
MIN_HEIGHT_M = 3.0


@dataclass
class BBox:
    """Bounding box in (south, west, north, east) order, WGS84."""
    south: float
    west: float
    north: float
    east: float

    def to_overpass(self) -> str:
        return f"{self.south},{self.west},{self.north},{self.east}"

    def to_list(self) -> list[float]:
        return [self.south, self.west, self.north, self.east]

    @classmethod
    def from_string(cls, s: str) -> "BBox":
        """Parse 'south,west,north,east' string."""
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"BBox needs 4 values (south,west,north,east), got {len(parts)}")
        return cls(*parts)

    @classmethod
    def valletta(cls) -> "BBox":
        return cls(*VALLETTA_BBOX)


def _build_query(bbox: BBox) -> str:
    """Build Overpass QL query for buildings within a bounding box."""
    bb = bbox.to_overpass()
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


def _parse_height(tags: dict[str, str]) -> float:
    """Extract or estimate building height in meters from OSM tags."""
    # Explicit height tag (may include unit suffix)
    raw_height = tags.get("height", "")
    if raw_height:
        try:
            # Strip common suffixes: "12 m", "12m", "40 ft"
            cleaned = raw_height.strip().lower()
            if cleaned.endswith("ft"):
                return float(cleaned.replace("ft", "").strip()) * 0.3048
            return float(cleaned.replace("m", "").strip())
        except (ValueError, TypeError):
            pass

    # Estimate from levels
    levels_str = tags.get("building:levels", tags.get("levels", ""))
    if levels_str:
        try:
            levels = int(float(levels_str))
            return max(levels * METERS_PER_LEVEL, MIN_HEIGHT_M)
        except (ValueError, TypeError):
            pass

    return DEFAULT_HEIGHT_M


def _extract_material(tags: dict[str, str]) -> Optional[str]:
    """Extract facade/building material from OSM tags."""
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
    return None


def _nodes_to_coords(way_nodes: list[int], node_map: dict[int, tuple[float, float]]) -> list[list[float]]:
    """Convert OSM node IDs to [lon, lat] coordinate ring."""
    coords = []
    for nid in way_nodes:
        if nid in node_map:
            lat, lon = node_map[nid]
            coords.append([lon, lat])
    # Close the ring if not already closed
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _color_from_material(material: Optional[str]) -> str:
    """Map material name to a hex color for MapLibre extrusion rendering."""
    colors = {
        "limestone": "#d4c9a8",
        "sandstone": "#d4b896",
        "stone": "#a0a090",
        "concrete": "#b0b0b0",
        "brick": "#a0522d",
        "glass": "#e0f0ff",
        "metal": "#c0c0c0",
        "wood": "#b87333",
        "plaster": "#f5f0e8",
        "stucco": "#e8dcc8",
        "render": "#e8e0d8",
        "marble": "#f0f0f0",
    }
    if material:
        for key, color in colors.items():
            if key in material:
                return color
    # Valletta default: limestone
    return "#d4c9a8"


def _compute_centroid(coords: list[list[float]]) -> tuple[float, float]:
    """Compute centroid of a coordinate ring. Returns (lon, lat)."""
    if not coords:
        return (0.0, 0.0)
    ring = coords[:-1] if coords[0] == coords[-1] and len(coords) > 1 else coords
    lon = sum(c[0] for c in ring) / len(ring)
    lat = sum(c[1] for c in ring) / len(ring)
    return (lon, lat)


def _parse_overpass_response(data: dict) -> dict[str, Any]:
    """
    Parse raw Overpass JSON into a GeoJSON FeatureCollection.

    Returns a dict compatible with MapLibre fill-extrusion-layer source.
    """
    elements = data.get("elements", [])

    # Build node coordinate lookup
    node_map: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    relations: list[dict] = []

    for elem in elements:
        etype = elem.get("type")
        if etype == "node":
            node_map[elem["id"]] = (elem["lat"], elem["lon"])
        elif etype == "way":
            ways.append(elem)
        elif etype == "relation":
            relations.append(elem)

    # Build way lookup for relation members
    way_map: dict[int, list[int]] = {}
    for w in ways:
        if "nodes" in w:
            way_map[w["id"]] = w["nodes"]

    features = []

    # Process standalone building ways
    for w in ways:
        tags = w.get("tags", {})
        if "building" not in tags:
            continue

        nodes = w.get("nodes", [])
        coords = _nodes_to_coords(nodes, node_map)
        if len(coords) < 4:
            continue

        height = _parse_height(tags)
        material = _extract_material(tags)
        centroid = _compute_centroid(coords)

        feature = {
            "type": "Feature",
            "id": w["id"],
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "osm_id": w["id"],
                "osm_type": "way",
                "building": tags.get("building", "yes"),
                "name": tags.get("name", ""),
                "height": height,
                "min_height": _parse_min_height(tags),
                "levels": _safe_int(tags.get("building:levels", "")),
                "material": material,
                "roof_shape": tags.get("roof:shape", "flat"),
                "roof_material": tags.get("roof:material"),
                "color": _color_from_material(material),
                "centroid_lon": centroid[0],
                "centroid_lat": centroid[1],
            },
        }
        features.append(feature)

    # Process multipolygon relations (complex buildings)
    for rel in relations:
        tags = rel.get("tags", {})
        if "building" not in tags:
            continue

        members = rel.get("members", [])
        outer_rings = []
        inner_rings = []

        for member in members:
            if member.get("type") != "way":
                continue
            role = member.get("role", "outer")
            way_nodes = way_map.get(member["ref"], [])
            coords = _nodes_to_coords(way_nodes, node_map)
            if len(coords) < 4:
                continue
            if role == "inner":
                inner_rings.append(coords)
            else:
                outer_rings.append(coords)

        if not outer_rings:
            continue

        height = _parse_height(tags)
        material = _extract_material(tags)

        # Build polygon coordinates: first outer ring, then inner rings (holes)
        polygon_coords = [outer_rings[0]] + inner_rings
        centroid = _compute_centroid(outer_rings[0])

        feature = {
            "type": "Feature",
            "id": rel["id"],
            "geometry": {
                "type": "Polygon",
                "coordinates": polygon_coords,
            },
            "properties": {
                "osm_id": rel["id"],
                "osm_type": "relation",
                "building": tags.get("building", "yes"),
                "name": tags.get("name", ""),
                "height": height,
                "min_height": _parse_min_height(tags),
                "levels": _safe_int(tags.get("building:levels", "")),
                "material": material,
                "roof_shape": tags.get("roof:shape", "flat"),
                "roof_material": tags.get("roof:material"),
                "color": _color_from_material(material),
                "centroid_lon": centroid[0],
                "centroid_lat": centroid[1],
            },
        }
        features.append(feature)

        # Additional outer rings as separate features (rare but possible)
        for extra_ring in outer_rings[1:]:
            centroid = _compute_centroid(extra_ring)
            extra_feature = {
                "type": "Feature",
                "id": rel["id"],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [extra_ring],
                },
                "properties": {
                    "osm_id": rel["id"],
                    "osm_type": "relation",
                    "building": tags.get("building", "yes"),
                    "name": tags.get("name", ""),
                    "height": height,
                    "min_height": _parse_min_height(tags),
                    "levels": _safe_int(tags.get("building:levels", "")),
                    "material": material,
                    "roof_shape": tags.get("roof:shape", "flat"),
                    "roof_material": tags.get("roof:material"),
                    "color": _color_from_material(material),
                    "centroid_lon": centroid[0],
                    "centroid_lat": centroid[1],
                },
            }
            features.append(extra_feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "count": len(features),
        },
    }

    logger.info("Parsed %d building features from Overpass response", len(features))
    return geojson


def _parse_min_height(tags: dict[str, str]) -> float:
    """Parse min_height tag for building parts."""
    raw = tags.get("min_height", "0")
    try:
        return float(raw.replace("m", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> Optional[int]:
    """Parse an integer, returning None on failure."""
    if not val:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_buildings(
    bbox: Optional[BBox] = None,
    timeout: float = OVERPASS_TIMEOUT,
) -> dict[str, Any]:
    """
    Fetch building footprints from OSM Overpass API for a bounding box.

    Args:
        bbox: Bounding box (south, west, north, east). Defaults to Valletta, Malta.
        timeout: HTTP request timeout in seconds.

    Returns:
        GeoJSON FeatureCollection with 3D extrusion properties per building:
            - height (meters)
            - min_height (meters)
            - levels (integer or None)
            - material (string or None)
            - color (hex string for MapLibre)
            - centroid_lon, centroid_lat
    """
    if bbox is None:
        bbox = BBox.valletta()

    query = _build_query(bbox)
    logger.info("Fetching buildings for bbox %s", bbox.to_overpass())

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    return _parse_overpass_response(data)


def fetch_buildings_sync(
    bbox: Optional[BBox] = None,
    timeout: float = OVERPASS_TIMEOUT,
) -> dict[str, Any]:
    """Synchronous version of fetch_buildings for scripts and testing."""
    if bbox is None:
        bbox = BBox.valletta()

    query = _build_query(bbox)
    logger.info("Fetching buildings (sync) for bbox %s", bbox.to_overpass())

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    return _parse_overpass_response(data)
