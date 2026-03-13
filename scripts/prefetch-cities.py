#!/usr/bin/env python3
"""
PlanO City Data Pre-fetcher.

Downloads and caches OSM building geometry, land use, road networks, and parks
for all PlanO cities via the Overpass API. Outputs GeoJSON files and statistics
per city, ready for the Rasta geometry pipeline.

Usage:
    python3 prefetch-cities.py --all              # Fetch all cities
    python3 prefetch-cities.py --city valletta     # Fetch one city
    python3 prefetch-cities.py --city paris --city monaco  # Fetch several
    python3 prefetch-cities.py --list              # List available cities

Requires: requests (stdlib json, os, time, argparse, math, pathlib used otherwise)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# City definitions — keep in sync with frontend city list
# bbox order: [south, west, north, east]
# ---------------------------------------------------------------------------

PLANO_CITIES = [
    # Malta
    {"id": "valletta", "name": "Valletta", "island": "Malta", "bbox": [35.893, 14.506, 35.905, 14.522]},
    {"id": "sliema", "name": "Sliema", "island": "Malta", "bbox": [35.907, 14.495, 35.920, 14.510]},
    {"id": "st-julians", "name": "St Julian's", "island": "Malta", "bbox": [35.912, 14.480, 35.930, 14.500]},
    {"id": "three-cities", "name": "Three Cities", "island": "Malta", "bbox": [35.880, 14.510, 35.895, 14.535]},
    {"id": "birkirkara", "name": "Birkirkara", "island": "Malta", "bbox": [35.888, 14.450, 35.905, 14.475]},
    {"id": "mosta", "name": "Mosta", "island": "Malta", "bbox": [35.900, 14.415, 35.920, 14.440]},
    {"id": "qormi", "name": "Qormi", "island": "Malta", "bbox": [35.868, 14.460, 35.885, 14.485]},
    {"id": "naxxar", "name": "Naxxar", "island": "Malta", "bbox": [35.905, 14.430, 35.925, 14.458]},
    {"id": "rabat-malta", "name": "Rabat", "island": "Malta", "bbox": [35.874, 14.388, 35.893, 14.412]},
    {"id": "mdina", "name": "Mdina", "island": "Malta", "bbox": [35.884, 14.399, 35.890, 14.407]},
    {"id": "marsaskala", "name": "Marsaskala", "island": "Malta", "bbox": [35.852, 14.555, 35.872, 14.578]},
    {"id": "mellieha", "name": "Mellieha", "island": "Malta", "bbox": [35.945, 14.348, 35.967, 14.378]},
    {"id": "swieqi", "name": "Swieqi", "island": "Malta", "bbox": [35.917, 14.472, 35.932, 14.490]},
    # Gozo
    {"id": "victoria-gozo", "name": "Victoria (Rabat)", "island": "Gozo", "bbox": [36.036, 14.230, 36.052, 14.250]},
    {"id": "xlendi", "name": "Xlendi", "island": "Gozo", "bbox": [36.023, 14.210, 36.033, 14.222]},
    {"id": "marsalforn", "name": "Marsalforn", "island": "Gozo", "bbox": [36.063, 14.245, 36.077, 14.265]},
    {"id": "nadur", "name": "Nadur", "island": "Gozo", "bbox": [36.028, 14.282, 36.046, 14.303]},
    {"id": "sannat", "name": "Sannat", "island": "Gozo", "bbox": [36.014, 14.233, 36.030, 14.253]},
    {"id": "gharb", "name": "Gharb", "island": "Gozo", "bbox": [36.050, 14.198, 36.068, 14.218]},
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 180  # seconds — generous for large cities
REQUEST_DELAY = 5.0  # seconds between Overpass requests (rate limit)
MAX_RETRIES = 4
INITIAL_BACKOFF = 10.0  # seconds for first retry on 429/503

METERS_PER_LEVEL = 3.2
DEFAULT_HEIGHT_M = 9.6
MIN_HEIGHT_M = 3.0

# Paris sub-grid: split bbox into NxN tiles to avoid Overpass timeout
PARIS_GRID_COLS = 3
PARIS_GRID_ROWS = 3

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "cities"
DATA_ROOT = _DEFAULT_DATA_ROOT

# ---------------------------------------------------------------------------
# Overpass query builders
# ---------------------------------------------------------------------------


def _bbox_str(bbox: list[float]) -> str:
    """Format bbox [south, west, north, east] for Overpass QL."""
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"


def _query_buildings(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
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


def _query_landuse(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["landuse"]({bb});
  relation["landuse"]({bb});
);
out body;
>;
out skel qt;
"""


def _query_roads(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street|pedestrian|service|unclassified)$"]({bb});
);
out body;
>;
out skel qt;
"""


def _query_parks(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["leisure"~"^(park|garden|playground|nature_reserve)$"]({bb});
  relation["leisure"~"^(park|garden|playground|nature_reserve)$"]({bb});
  way["landuse"="recreation_ground"]({bb});
  way["natural"="wood"]({bb});
);
out body;
>;
out skel qt;
"""


# ---------------------------------------------------------------------------
# Overpass HTTP client with retry and backoff
# ---------------------------------------------------------------------------


def _overpass_request(query: str, label: str = "") -> dict:
    """
    Send a query to the Overpass API with exponential backoff on 429/503.

    Returns the parsed JSON response.
    Raises SystemExit on persistent failure.
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"Accept": "application/json"},
                timeout=OVERPASS_TIMEOUT + 30,  # HTTP timeout > Overpass timeout
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 503, 504):
                wait = backoff * (2 ** (attempt - 1))
                _log(f"  [{label}] HTTP {resp.status_code} — retrying in {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            # Other errors: fail immediately
            _log(f"  [{label}] HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()

        except requests.exceptions.Timeout:
            wait = backoff * (2 ** (attempt - 1))
            _log(f"  [{label}] Timeout — retrying in {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        except requests.exceptions.ConnectionError as exc:
            wait = backoff * (2 ** (attempt - 1))
            _log(f"  [{label}] Connection error: {exc} — retrying in {wait:.0f}s")
            time.sleep(wait)
            continue

    _log(f"  [{label}] FAILED after {MAX_RETRIES} attempts. Skipping.")
    return {"elements": []}


# ---------------------------------------------------------------------------
# Overpass response parsers -> GeoJSON
# ---------------------------------------------------------------------------


def _build_node_map(elements: list[dict]) -> dict[int, tuple[float, float]]:
    """Build { node_id: (lat, lon) } lookup from Overpass elements."""
    node_map = {}
    for elem in elements:
        if elem.get("type") == "node":
            node_map[elem["id"]] = (elem["lat"], elem["lon"])
    return node_map


def _nodes_to_coords(node_ids: list[int], node_map: dict[int, tuple[float, float]]) -> list[list[float]]:
    """Convert OSM node IDs to [lon, lat] coordinate ring (GeoJSON order)."""
    coords = []
    for nid in node_ids:
        if nid in node_map:
            lat, lon = node_map[nid]
            coords.append([lon, lat])
    # Close ring
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _parse_height(tags: dict[str, str]) -> float:
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
            return max(int(float(levels_str)) * METERS_PER_LEVEL, MIN_HEIGHT_M)
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


def _safe_int(val: str) -> Optional[int]:
    if not val:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _polygon_area_sq_deg(coords: list[list[float]]) -> float:
    """Shoelace formula for polygon area in square degrees (approximate)."""
    n = len(coords)
    if n < 4:
        return 0.0
    area = 0.0
    for i in range(n - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _sq_deg_to_sq_m(area_deg: float, lat: float) -> float:
    """Convert area in square degrees to approximate square meters."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return area_deg * m_per_deg_lat * m_per_deg_lon


def _build_way_map(elements: list[dict]) -> dict[int, list[int]]:
    """Build { way_id: [node_ids] } lookup from Overpass elements."""
    way_map = {}
    for elem in elements:
        if elem.get("type") == "way" and "nodes" in elem:
            way_map[elem["id"]] = elem["nodes"]
    return way_map


def _extract_polygon_features(
    elements: list[dict],
    tag_key: str,
    property_extractor: callable,
) -> list[dict]:
    """
    Generic extractor for polygon features (ways + relations) from Overpass response.

    tag_key: the OSM tag that identifies relevant elements (e.g. "building", "landuse")
    property_extractor: callable(tags, osm_id, osm_type) -> dict of GeoJSON properties
    """
    node_map = _build_node_map(elements)
    way_map = _build_way_map(elements)
    features = []

    # Ways
    for elem in elements:
        if elem.get("type") != "way":
            continue
        tags = elem.get("tags", {})
        if tag_key not in tags:
            continue
        nodes = elem.get("nodes", [])
        coords = _nodes_to_coords(nodes, node_map)
        if len(coords) < 4:
            continue
        props = property_extractor(tags, elem["id"], "way")
        features.append({
            "type": "Feature",
            "id": elem["id"],
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": props,
        })

    # Relations (multipolygon)
    for elem in elements:
        if elem.get("type") != "relation":
            continue
        tags = elem.get("tags", {})
        if tag_key not in tags:
            continue
        members = elem.get("members", [])
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

        polygon_coords = [outer_rings[0]] + inner_rings
        props = property_extractor(tags, elem["id"], "relation")
        features.append({
            "type": "Feature",
            "id": elem["id"],
            "geometry": {"type": "Polygon", "coordinates": polygon_coords},
            "properties": props,
        })
        # Extra outer rings as separate features
        for extra in outer_rings[1:]:
            features.append({
                "type": "Feature",
                "id": elem["id"],
                "geometry": {"type": "Polygon", "coordinates": [extra]},
                "properties": props,
            })

    return features


def _extract_line_features(
    elements: list[dict],
    tag_key: str,
    property_extractor: callable,
) -> list[dict]:
    """Extract LineString features (ways) from Overpass response."""
    node_map = _build_node_map(elements)
    features = []

    for elem in elements:
        if elem.get("type") != "way":
            continue
        tags = elem.get("tags", {})
        if tag_key not in tags:
            continue
        nodes = elem.get("nodes", [])
        coords = []
        for nid in nodes:
            if nid in node_map:
                lat, lon = node_map[nid]
                coords.append([lon, lat])
        if len(coords) < 2:
            continue
        props = property_extractor(tags, elem["id"], "way")
        features.append({
            "type": "Feature",
            "id": elem["id"],
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    return features


# ---------------------------------------------------------------------------
# Property extractors per layer
# ---------------------------------------------------------------------------


def _building_props(tags: dict, osm_id: int, osm_type: str) -> dict:
    height = _parse_height(tags)
    material = _extract_material(tags)
    return {
        "osm_id": osm_id,
        "osm_type": osm_type,
        "building": tags.get("building", "yes"),
        "name": tags.get("name", ""),
        "height": height,
        "levels": _safe_int(tags.get("building:levels", "")),
        "material": material,
        "roof_shape": tags.get("roof:shape", "flat"),
        "addr_street": tags.get("addr:street", ""),
        "addr_housenumber": tags.get("addr:housenumber", ""),
        "addr_city": tags.get("addr:city", ""),
    }


def _landuse_props(tags: dict, osm_id: int, osm_type: str) -> dict:
    return {
        "osm_id": osm_id,
        "osm_type": osm_type,
        "landuse": tags.get("landuse", ""),
        "name": tags.get("name", ""),
    }


def _road_props(tags: dict, osm_id: int, osm_type: str) -> dict:
    return {
        "osm_id": osm_id,
        "osm_type": osm_type,
        "highway": tags.get("highway", ""),
        "name": tags.get("name", ""),
        "surface": tags.get("surface", ""),
        "lanes": _safe_int(tags.get("lanes", "")),
        "oneway": tags.get("oneway", ""),
        "maxspeed": tags.get("maxspeed", ""),
    }


def _park_props(tags: dict, osm_id: int, osm_type: str) -> dict:
    return {
        "osm_id": osm_id,
        "osm_type": osm_type,
        "leisure": tags.get("leisure", ""),
        "landuse": tags.get("landuse", ""),
        "natural": tags.get("natural", ""),
        "name": tags.get("name", ""),
    }


# ---------------------------------------------------------------------------
# Bbox subdivision for large cities (Paris)
# ---------------------------------------------------------------------------


def _split_bbox(bbox: list[float], rows: int, cols: int) -> list[list[float]]:
    """Split a bounding box into a grid of sub-bboxes."""
    south, west, north, east = bbox
    lat_step = (north - south) / rows
    lon_step = (east - west) / cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = [
                south + r * lat_step,
                west + c * lon_step,
                south + (r + 1) * lat_step,
                west + (c + 1) * lon_step,
            ]
            tiles.append(tile)
    return tiles


def _needs_subdivision(city: dict) -> bool:
    """Determine if a city bbox is large enough to require subdivision."""
    bbox = city["bbox"]
    lat_span = bbox[2] - bbox[0]
    lon_span = bbox[3] - bbox[1]
    # Paris-sized or bigger: > 0.05 deg in both axes
    return lat_span > 0.06 and lon_span > 0.15


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _compute_stats(buildings_geojson: dict, city: dict) -> dict:
    """Compute summary statistics from a buildings GeoJSON FeatureCollection."""
    features = buildings_geojson.get("features", [])
    building_count = len(features)

    heights = []
    materials = {}
    building_types = {}
    total_area_sqm = 0.0
    named_count = 0
    addressed_count = 0

    bbox = city["bbox"]
    center_lat = (bbox[0] + bbox[2]) / 2.0

    for feat in features:
        props = feat.get("properties", {})
        h = props.get("height", DEFAULT_HEIGHT_M)
        heights.append(h)

        mat = props.get("material") or "unknown"
        materials[mat] = materials.get(mat, 0) + 1

        btype = props.get("building", "yes")
        building_types[btype] = building_types.get(btype, 0) + 1

        if props.get("name"):
            named_count += 1
        if props.get("addr_street") or props.get("addr_housenumber"):
            addressed_count += 1

        # Approximate footprint area
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [[]])
        if coords and coords[0]:
            area_deg = _polygon_area_sq_deg(coords[0])
            total_area_sqm += _sq_deg_to_sq_m(area_deg, center_lat)

    avg_height = sum(heights) / len(heights) if heights else 0.0
    max_height = max(heights) if heights else 0.0
    min_height = min(heights) if heights else 0.0

    # Sort distributions by count descending
    materials_sorted = dict(sorted(materials.items(), key=lambda x: -x[1]))
    types_sorted = dict(sorted(building_types.items(), key=lambda x: -x[1]))

    return {
        "city_id": city["id"],
        "city_name": city["name"],
        "country": city["country"],
        "bbox": city["bbox"],
        "building_count": building_count,
        "total_footprint_area_sqm": round(total_area_sqm, 1),
        "named_buildings": named_count,
        "addressed_buildings": addressed_count,
        "height_avg_m": round(avg_height, 2),
        "height_max_m": round(max_height, 2),
        "height_min_m": round(min_height, 2),
        "material_distribution": materials_sorted,
        "building_type_distribution": types_sorted,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Layer fetch orchestration
# ---------------------------------------------------------------------------


def _fetch_layer_polygon(
    bbox: list[float],
    query_fn: callable,
    tag_key: str,
    prop_fn: callable,
    label: str,
    subdivide: bool = False,
) -> dict:
    """Fetch a polygon layer, optionally with bbox subdivision."""
    all_features = []

    if subdivide:
        tiles = _split_bbox(bbox, PARIS_GRID_ROWS, PARIS_GRID_COLS)
        _log(f"    Subdividing into {len(tiles)} tiles for {label}")
        for i, tile in enumerate(tiles):
            _log(f"    Tile {i + 1}/{len(tiles)} ...")
            query = query_fn(tile)
            data = _overpass_request(query, label=f"{label}-tile{i + 1}")
            features = _extract_polygon_features(data.get("elements", []), tag_key, prop_fn)
            all_features.extend(features)
            if i < len(tiles) - 1:
                time.sleep(REQUEST_DELAY)
    else:
        query = query_fn(bbox)
        data = _overpass_request(query, label=label)
        all_features = _extract_polygon_features(data.get("elements", []), tag_key, prop_fn)

    # Deduplicate by osm_id (tiles may overlap at edges)
    seen_ids = set()
    deduped = []
    for feat in all_features:
        fid = feat.get("id")
        if fid not in seen_ids:
            seen_ids.add(fid)
            deduped.append(feat)

    return {
        "type": "FeatureCollection",
        "features": deduped,
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "count": len(deduped),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def _fetch_layer_line(
    bbox: list[float],
    query_fn: callable,
    tag_key: str,
    prop_fn: callable,
    label: str,
    subdivide: bool = False,
) -> dict:
    """Fetch a line layer, optionally with bbox subdivision."""
    all_features = []

    if subdivide:
        tiles = _split_bbox(bbox, PARIS_GRID_ROWS, PARIS_GRID_COLS)
        _log(f"    Subdividing into {len(tiles)} tiles for {label}")
        for i, tile in enumerate(tiles):
            _log(f"    Tile {i + 1}/{len(tiles)} ...")
            query = query_fn(tile)
            data = _overpass_request(query, label=f"{label}-tile{i + 1}")
            features = _extract_line_features(data.get("elements", []), tag_key, prop_fn)
            all_features.extend(features)
            if i < len(tiles) - 1:
                time.sleep(REQUEST_DELAY)
    else:
        query = query_fn(bbox)
        data = _overpass_request(query, label=label)
        all_features = _extract_line_features(data.get("elements", []), tag_key, prop_fn)

    seen_ids = set()
    deduped = []
    for feat in all_features:
        fid = feat.get("id")
        if fid not in seen_ids:
            seen_ids.add(fid)
            deduped.append(feat)

    return {
        "type": "FeatureCollection",
        "features": deduped,
        "metadata": {
            "source": "OpenStreetMap via Overpass API",
            "count": len(deduped),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


# ---------------------------------------------------------------------------
# Per-city fetch
# ---------------------------------------------------------------------------


def fetch_city(city: dict) -> None:
    """Fetch all layers for a single city and write to disk."""
    city_id = city["id"]
    city_name = city["name"]
    bbox = city["bbox"]
    subdivide = _needs_subdivision(city)

    out_dir = DATA_ROOT / city_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"\n{'=' * 60}")
    _log(f"Fetching: {city_name} ({city['country']}) — bbox {bbox}")
    if subdivide:
        _log(f"  Large city detected — will subdivide queries into {PARIS_GRID_ROWS * PARIS_GRID_COLS} tiles")
    _log(f"{'=' * 60}")

    # 1. Buildings
    _log(f"  [1/4] Buildings ...")
    t0 = time.time()
    buildings = _fetch_layer_polygon(bbox, _query_buildings, "building", _building_props, f"{city_id}/buildings", subdivide)
    elapsed = time.time() - t0
    _log(f"         {buildings['metadata']['count']} buildings in {elapsed:.1f}s")
    _write_geojson(out_dir / "buildings.geojson", buildings)
    time.sleep(REQUEST_DELAY)

    # 2. Land use
    _log(f"  [2/4] Land use ...")
    t0 = time.time()
    landuse = _fetch_layer_polygon(bbox, _query_landuse, "landuse", _landuse_props, f"{city_id}/landuse", subdivide)
    elapsed = time.time() - t0
    _log(f"         {landuse['metadata']['count']} zones in {elapsed:.1f}s")
    _write_geojson(out_dir / "landuse.geojson", landuse)
    time.sleep(REQUEST_DELAY)

    # 3. Roads
    _log(f"  [3/4] Roads ...")
    t0 = time.time()
    roads = _fetch_layer_line(bbox, _query_roads, "highway", _road_props, f"{city_id}/roads", subdivide)
    elapsed = time.time() - t0
    _log(f"         {roads['metadata']['count']} road segments in {elapsed:.1f}s")
    _write_geojson(out_dir / "roads.geojson", roads)
    time.sleep(REQUEST_DELAY)

    # 4. Parks
    _log(f"  [4/4] Parks ...")
    t0 = time.time()
    parks = _fetch_layer_polygon(bbox, _query_parks, "leisure", _park_props, f"{city_id}/parks", subdivide)
    # Also capture landuse=recreation_ground and natural=wood that use different tag keys
    # They are included in the query but may have leisure absent; re-extract with broader matching
    parks_data_raw = _overpass_request(_query_parks(bbox), label=f"{city_id}/parks-extra") if not subdivide else {"elements": []}
    extra_landuse = _extract_polygon_features(parks_data_raw.get("elements", []), "landuse", _park_props)
    extra_natural = _extract_polygon_features(parks_data_raw.get("elements", []), "natural", _park_props)
    # Merge without duplicates
    seen = {f.get("id") for f in parks["features"]}
    for feat in extra_landuse + extra_natural:
        if feat.get("id") not in seen:
            seen.add(feat.get("id"))
            parks["features"].append(feat)
    parks["metadata"]["count"] = len(parks["features"])
    elapsed = time.time() - t0
    _log(f"         {parks['metadata']['count']} green areas in {elapsed:.1f}s")
    _write_geojson(out_dir / "parks.geojson", parks)

    # 5. Stats
    stats = _compute_stats(buildings, city)
    stats["layers"] = {
        "buildings": buildings["metadata"]["count"],
        "landuse": landuse["metadata"]["count"],
        "roads": roads["metadata"]["count"],
        "parks": parks["metadata"]["count"],
    }
    _write_json(out_dir / "stats.json", stats)
    _log(f"\n  Summary for {city_name}:")
    _log(f"    Buildings:  {stats['building_count']}")
    _log(f"    Footprint:  {stats['total_footprint_area_sqm']:,.0f} m2")
    _log(f"    Avg height: {stats['height_avg_m']} m")
    _log(f"    Max height: {stats['height_max_m']} m")
    _log(f"    Materials:  {json.dumps(dict(list(stats['material_distribution'].items())[:5]))}")
    _log(f"    Saved to:   {out_dir}/")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_geojson(path: Path, data: dict) -> None:
    """Write GeoJSON with compact but readable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = path.stat().st_size / (1024 * 1024)
    _log(f"         Wrote {path.name} ({size_mb:.2f} MB)")


def _write_json(path: Path, data: dict) -> None:
    """Write JSON with indentation for readability."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _log(msg: str) -> None:
    """Print with flush for immediate output in pipelines."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _get_city_by_id(city_id: str) -> Optional[dict]:
    for city in PLANO_CITIES:
        if city["id"] == city_id:
            return city
    return None


def main() -> None:
    global DATA_ROOT

    parser = argparse.ArgumentParser(
        description="PlanO city OSM data pre-fetcher. Downloads building geometry, land use, roads, and parks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                     Fetch all 9 cities
  %(prog)s --city valletta           Fetch Valletta only
  %(prog)s --city monaco --city nice Fetch Monaco and Nice
  %(prog)s --list                    Show available cities
        """,
    )
    parser.add_argument("--all", action="store_true", help="Fetch all PlanO cities")
    parser.add_argument("--city", action="append", default=[], metavar="ID", help="City ID to fetch (repeatable)")
    parser.add_argument("--list", action="store_true", help="List available cities and exit")
    parser.add_argument("--data-dir", type=str, default=None, help=f"Output directory (default: {DATA_ROOT})")

    args = parser.parse_args()

    if args.data_dir:
        DATA_ROOT = Path(args.data_dir)

    if args.list:
        _log("Available PlanO cities:")
        _log(f"  {'ID':<15} {'Name':<20} {'Country':<8} {'BBox'}")
        _log(f"  {'-' * 15} {'-' * 20} {'-' * 8} {'-' * 40}")
        for city in PLANO_CITIES:
            _log(f"  {city['id']:<15} {city['name']:<20} {city['country']:<8} {city['bbox']}")
        return

    if not args.all and not args.city:
        parser.error("Specify --all or --city <id>. Use --list to see available cities.")

    # Resolve city list
    if args.all:
        cities = PLANO_CITIES
    else:
        cities = []
        for cid in args.city:
            city = _get_city_by_id(cid)
            if city is None:
                available = ", ".join(c["id"] for c in PLANO_CITIES)
                parser.error(f"Unknown city '{cid}'. Available: {available}")
            cities.append(city)

    _log(f"PlanO City Pre-fetcher")
    _log(f"Output directory: {DATA_ROOT}")
    _log(f"Cities to fetch:  {len(cities)} — {', '.join(c['name'] for c in cities)}")
    _log(f"Rate limit:       {REQUEST_DELAY}s between requests")

    # Estimate time: 4 layers per city, ~5s delay each, +query time
    # Subdivided cities: 4 layers * 9 tiles * 5s = ~180s per layer
    total_requests = 0
    for city in cities:
        if _needs_subdivision(city):
            total_requests += 4 * PARIS_GRID_ROWS * PARIS_GRID_COLS
        else:
            total_requests += 4
    est_minutes = (total_requests * (REQUEST_DELAY + 5)) / 60  # rough: 5s avg query + 5s delay
    _log(f"Estimated time:   ~{est_minutes:.0f} minutes ({total_requests} API requests)")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    failed = 0

    for i, city in enumerate(cities):
        try:
            fetch_city(city)
            completed += 1
        except KeyboardInterrupt:
            _log("\nInterrupted by user.")
            break
        except Exception as exc:
            _log(f"\n  ERROR fetching {city['name']}: {exc}")
            failed += 1

        # Progress
        done = i + 1
        elapsed = time.time() - t_start
        if done < len(cities):
            avg_per_city = elapsed / done
            remaining = (len(cities) - done) * avg_per_city
            _log(f"\n  Progress: {done}/{len(cities)} cities — ~{remaining / 60:.0f} min remaining")

    elapsed_total = time.time() - t_start
    _log(f"\n{'=' * 60}")
    _log(f"Done. {completed} cities fetched, {failed} failed.")
    _log(f"Total time: {elapsed_total / 60:.1f} minutes")
    _log(f"Data saved to: {DATA_ROOT}/")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
