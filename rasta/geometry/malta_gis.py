"""
Malta Spatial Data Infrastructure (MSDI) and PA MapServer connector.

Fetches GIS layers from Malta government WMS/WFS services and the
Planning Authority MapServer for use in the PlanO building intelligence
system.  Enriches building GeoJSON with planning zone, heritage, and
protected area metadata.

Layer sources:
    - MSDI WFS/WMS: https://msdi.data.gov.mt/geoserver/wms|wfs
    - PA MapServer:  https://pamapserver.pa.org.mt/arcgis/rest/services

Only depends on `requests` + stdlib.  No geopandas, no shapely.
Spatial intersection uses simple bbox overlap (sufficient for enrichment).

Usage:
    python -m rasta.geometry.malta_gis --discover
"""

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

MSDI_WMS = "https://msdi.data.gov.mt/geoserver/wms"
MSDI_WFS = "https://msdi.data.gov.mt/geoserver/wfs"

PA_MAPSERVER_BASE = "https://pamapserver.pa.org.mt/arcgis/rest/services"

# Default CRS for all requests (WGS84 geographic)
DEFAULT_CRS = "EPSG:4326"

# HTTP settings — Malta government services can be slow or flaky
HTTP_TIMEOUT = 60
HTTP_RETRIES = 2
HTTP_BACKOFF = 5.0

# ---------------------------------------------------------------------------
# Well-known layer names
#
# These may change if MSDI reorganises.  Use discover_layers() to get the
# current list and update these constants.
# ---------------------------------------------------------------------------

# MSDI WFS layers (namespace:layername)
LAYER_DEVELOPMENT_ZONES = "pa:development_zone_boundaries"
LAYER_SCHEDULED_PROPERTIES = "pa:scheduled_properties"
LAYER_LAND_USE = "pa:land_use_zoning"
LAYER_PROTECTED_AREAS = "era:natura_2000"
LAYER_ROAD_CENTRELINES = "tms:road_centrelines"

# PA MapServer service/layer IDs (ArcGIS REST)
PA_SERVICE_DEVELOPMENT_ZONES = "PA/DevelopmentZoneBoundaries/MapServer/0"
PA_SERVICE_SCHEDULED = "PA/ScheduledProperties/MapServer/0"

# ---------------------------------------------------------------------------
# In-memory cache with 24-hour TTL
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL = 86400  # 24 hours


def _cache_key(prefix: str, bbox: Tuple[float, ...], layer: str) -> str:
    bbox_str = ",".join(f"{v:.5f}" for v in bbox)
    return f"{prefix}:{layer}:{bbox_str}"


def _cache_get(key: str) -> Optional[Any]:
    if key in _CACHE:
        ts, data = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            logger.debug("Cache hit: %s", key)
            return data
        del _CACHE[key]
    return None


def _cache_put(key: str, data: Any) -> None:
    if len(_CACHE) > 200:
        now = time.time()
        expired = [k for k, (ts, _) in _CACHE.items() if now - ts >= CACHE_TTL]
        for k in expired:
            del _CACHE[k]
    _CACHE[key] = (time.time(), data)


# ---------------------------------------------------------------------------
# HTTP helper with retry
# ---------------------------------------------------------------------------

def _http_get(
    url: str,
    params: Optional[Dict[str, str]] = None,
    retries: int = HTTP_RETRIES,
    backoff: float = HTTP_BACKOFF,
    timeout: int = HTTP_TIMEOUT,
) -> Optional[requests.Response]:
    """
    GET request with retry on transient errors.

    Returns the Response on success, or None on persistent failure.
    Malta government services frequently return 502/503 or simply time out.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "HTTP %d from %s — retry %d/%d in %.0fs",
                    resp.status_code, url, attempt, retries, wait,
                )
                time.sleep(wait)
                continue
            logger.error("HTTP %d from %s: %s", resp.status_code, url, resp.text[:200])
            return None
        except requests.exceptions.Timeout:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Timeout on %s — retry %d/%d in %.0fs", url, attempt, retries, wait)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Connection error on %s: %s — retry in %.0fs", url, exc, wait)
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return None

    logger.error("All %d attempts failed for %s", retries, url)
    return None


# ---------------------------------------------------------------------------
# WFS client
# ---------------------------------------------------------------------------

def _bbox_to_wfs(bbox: Tuple[float, float, float, float]) -> str:
    """Format bbox as WFS filter string: south,west,north,east."""
    south, west, north, east = bbox
    return f"{south},{west},{north},{east}"


def fetch_wfs_features(
    bbox: Tuple[float, float, float, float],
    layer_name: str,
    max_features: int = 1000,
    service_url: str = MSDI_WFS,
) -> Dict[str, Any]:
    """
    Fetch features from MSDI WFS as GeoJSON.

    Parameters
    ----------
    bbox : tuple
        (south, west, north, east) in WGS84 degrees.
    layer_name : str
        Fully qualified layer name, e.g. "pa:development_zone_boundaries".
    max_features : int
        Maximum features to return.
    service_url : str
        WFS endpoint URL.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.  Empty collection on failure.
    """
    cache_k = _cache_key("wfs", bbox, layer_name)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer_name,
        "outputFormat": "application/json",
        "srsName": DEFAULT_CRS,
        "bbox": f"{_bbox_to_wfs(bbox)},{DEFAULT_CRS}",
        "count": str(max_features),
    }

    logger.info("WFS GetFeature: %s bbox=%s", layer_name, bbox)
    resp = _http_get(service_url, params=params)

    if resp is None:
        logger.error("WFS request failed for layer %s", layer_name)
        return _empty_collection(layer_name, bbox)

    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        logger.error("WFS response is not valid JSON for layer %s", layer_name)
        return _empty_collection(layer_name, bbox)

    # Normalise to FeatureCollection if GeoServer returns something else
    if data.get("type") != "FeatureCollection":
        if "features" in data:
            data["type"] = "FeatureCollection"
        else:
            logger.warning("Unexpected WFS response structure for %s", layer_name)
            return _empty_collection(layer_name, bbox)

    result = {
        "type": "FeatureCollection",
        "features": data.get("features", []),
        "metadata": {
            "source": f"MSDI WFS ({layer_name})",
            "count": len(data.get("features", [])),
            "bbox": list(bbox),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    _cache_put(cache_k, result)
    logger.info("WFS returned %d features for %s", result["metadata"]["count"], layer_name)
    return result


def _empty_collection(
    source: str,
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {
            "source": source,
            "count": 0,
            "bbox": list(bbox),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": True,
        },
    }


# ---------------------------------------------------------------------------
# PA MapServer (ArcGIS REST) client
# ---------------------------------------------------------------------------

def _fetch_pa_mapserver(
    bbox: Tuple[float, float, float, float],
    service_path: str,
    max_features: int = 1000,
) -> Dict[str, Any]:
    """
    Query PA MapServer ArcGIS REST endpoint.

    The PA MapServer uses Esri's REST API.  We request GeoJSON output
    from the /query endpoint.

    Returns a GeoJSON FeatureCollection.
    """
    cache_k = _cache_key("pa", bbox, service_path)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached

    south, west, north, east = bbox
    url = f"{PA_MAPSERVER_BASE}/{service_path}/query"

    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": str(max_features),
    }

    logger.info("PA MapServer query: %s bbox=%s", service_path, bbox)
    resp = _http_get(url, params=params)

    if resp is None:
        logger.error("PA MapServer request failed for %s", service_path)
        return _empty_collection(f"PA MapServer ({service_path})", bbox)

    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        logger.error("PA MapServer response is not valid JSON for %s", service_path)
        return _empty_collection(f"PA MapServer ({service_path})", bbox)

    # ArcGIS sometimes wraps errors in a JSON object with an "error" key
    if "error" in data:
        logger.error(
            "PA MapServer error for %s: %s",
            service_path, data["error"].get("message", data["error"]),
        )
        return _empty_collection(f"PA MapServer ({service_path})", bbox)

    features = data.get("features", [])
    result = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": f"PA MapServer ({service_path})",
            "count": len(features),
            "bbox": list(bbox),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    _cache_put(cache_k, result)
    logger.info("PA MapServer returned %d features for %s", len(features), service_path)
    return result


# ---------------------------------------------------------------------------
# High-level fetch functions
# ---------------------------------------------------------------------------

def fetch_development_zones(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Fetch PA development zone boundaries for a bounding box.

    Tries PA MapServer first (usually more reliable), falls back to
    MSDI WFS if the PA endpoint is unavailable.
    """
    result = _fetch_pa_mapserver(bbox, PA_SERVICE_DEVELOPMENT_ZONES)
    if result["metadata"].get("error"):
        logger.info("PA MapServer failed for dev zones, falling back to MSDI WFS")
        result = fetch_wfs_features(bbox, LAYER_DEVELOPMENT_ZONES)
    return result


def fetch_scheduled_properties(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Fetch heritage-scheduled buildings and areas.

    Tries PA MapServer first, falls back to MSDI WFS.
    """
    result = _fetch_pa_mapserver(bbox, PA_SERVICE_SCHEDULED)
    if result["metadata"].get("error"):
        logger.info("PA MapServer failed for scheduled, falling back to MSDI WFS")
        result = fetch_wfs_features(bbox, LAYER_SCHEDULED_PROPERTIES)
    return result


def fetch_protected_areas(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """Fetch ERA Natura 2000 and other protected areas from MSDI WFS."""
    return fetch_wfs_features(bbox, LAYER_PROTECTED_AREAS)


def fetch_land_use_zones(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """Fetch land use zoning data from MSDI WFS."""
    return fetch_wfs_features(bbox, LAYER_LAND_USE)


def fetch_road_centrelines(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """Fetch road centreline geometry from MSDI WFS."""
    return fetch_wfs_features(bbox, LAYER_ROAD_CENTRELINES)


# ---------------------------------------------------------------------------
# Layer discovery (GetCapabilities)
# ---------------------------------------------------------------------------

def discover_layers(service_url: str = MSDI_WFS) -> List[Dict[str, str]]:
    """
    Parse WFS/WMS GetCapabilities to list available layer names.

    Parameters
    ----------
    service_url : str
        The WMS or WFS endpoint URL.

    Returns
    -------
    list of dict
        Each dict has keys: "name", "title", "abstract" (if available).
    """
    # Detect service type from URL
    is_wms = "wms" in service_url.lower()
    service_type = "WMS" if is_wms else "WFS"

    params = {
        "service": service_type,
        "request": "GetCapabilities",
    }

    logger.info("GetCapabilities: %s (%s)", service_url, service_type)
    resp = _http_get(service_url, params=params, timeout=30)

    if resp is None:
        logger.error("GetCapabilities failed for %s", service_url)
        return []

    return _parse_capabilities_xml(resp.text, is_wms)


def _parse_capabilities_xml(xml_text: str, is_wms: bool) -> List[Dict[str, str]]:
    """
    Extract layer names from a GetCapabilities XML response.

    Handles both WMS and WFS capabilities documents with or without
    XML namespaces.
    """
    layers: List[Dict[str, str]] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse capabilities XML: %s", exc)
        return []

    # WFS uses <FeatureType> elements; WMS uses <Layer> elements
    # Both can have arbitrary namespace prefixes, so search by local name
    if is_wms:
        tag_targets = ["Layer"]
    else:
        tag_targets = ["FeatureType"]

    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local not in tag_targets:
            continue

        name_elem = _find_child(elem, "Name")
        title_elem = _find_child(elem, "Title")
        abstract_elem = _find_child(elem, "Abstract")

        if name_elem is not None and name_elem.text:
            entry = {"name": name_elem.text.strip()}
            if title_elem is not None and title_elem.text:
                entry["title"] = title_elem.text.strip()
            if abstract_elem is not None and abstract_elem.text:
                entry["abstract"] = abstract_elem.text.strip()
            layers.append(entry)

    return layers


def _find_child(parent: ET.Element, local_name: str) -> Optional[ET.Element]:
    """Find a direct child element by local name, ignoring namespace."""
    for child in parent:
        child_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if child_local == local_name:
            return child
    return None


def discover_pa_mapserver_services() -> List[Dict[str, str]]:
    """
    List available services from the PA MapServer REST catalogue.

    Returns a list of dicts with "name" and "type" keys.
    """
    url = f"{PA_MAPSERVER_BASE}"
    params = {"f": "json"}

    resp = _http_get(url, params=params, timeout=30)
    if resp is None:
        logger.error("PA MapServer catalogue request failed")
        return []

    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        logger.error("PA MapServer catalogue is not valid JSON")
        return []

    services = []
    for svc in data.get("services", []):
        services.append({
            "name": svc.get("name", ""),
            "type": svc.get("type", ""),
        })

    # Also check folders for nested services
    for folder in data.get("folders", []):
        folder_url = f"{PA_MAPSERVER_BASE}/{folder}"
        folder_resp = _http_get(folder_url, params={"f": "json"}, timeout=30)
        if folder_resp is None:
            continue
        try:
            folder_data = folder_resp.json()
        except (ValueError, json.JSONDecodeError):
            continue
        for svc in folder_data.get("services", []):
            services.append({
                "name": svc.get("name", ""),
                "type": svc.get("type", ""),
            })

    return services


# ---------------------------------------------------------------------------
# Spatial utilities (bbox-only, no shapely)
# ---------------------------------------------------------------------------

def _feature_bbox(feature: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute the bounding box of a GeoJSON feature from its coordinates.

    Returns (south, west, north, east) or None if geometry is missing.
    """
    geom = feature.get("geometry")
    if not geom:
        return None

    coords = geom.get("coordinates")
    if not coords:
        return None

    # Flatten nested coordinate arrays to get all [lon, lat] pairs
    flat = _flatten_coords(coords)
    if not flat:
        return None

    lons = [p[0] for p in flat]
    lats = [p[1] for p in flat]

    return (min(lats), min(lons), max(lats), max(lons))


def _flatten_coords(coords: Any) -> List[List[float]]:
    """Recursively flatten nested coordinate arrays to a list of [lon, lat]."""
    if not coords:
        return []
    # Base case: a coordinate pair [lon, lat]
    if isinstance(coords[0], (int, float)):
        return [coords]
    # Recursive case
    result = []
    for item in coords:
        result.extend(_flatten_coords(item))
    return result


def _bboxes_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """
    Check if two bounding boxes overlap.

    Each bbox is (south, west, north, east).
    """
    a_south, a_west, a_north, a_east = a
    b_south, b_west, b_north, b_east = b

    if a_north < b_south or b_north < a_south:
        return False
    if a_east < b_west or b_east < a_west:
        return False
    return True


# ---------------------------------------------------------------------------
# Building enrichment
# ---------------------------------------------------------------------------

def enrich_buildings_with_planning(
    buildings_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Cross-reference buildings with planning zones, heritage status, and
    protected areas.

    Adds properties to each building feature:
        - in_development_zone (bool)
        - zone_type (str or None) — development zone category
        - heritage_scheduled (bool)
        - protected_area (bool)
        - protected_area_name (str or None)

    Uses simple bbox overlap for spatial intersection (no shapely).
    This is an approximation but sufficient for enrichment purposes.

    Parameters
    ----------
    buildings_geojson : dict
        GeoJSON FeatureCollection of buildings.
    bbox : tuple
        (south, west, north, east) bounding box to fetch planning data for.

    Returns
    -------
    dict
        The same FeatureCollection with enriched properties.
    """
    # Fetch all planning layers in parallel conceptually, but sequentially
    # to avoid hammering the services
    dev_zones = fetch_development_zones(bbox)
    scheduled = fetch_scheduled_properties(bbox)
    protected = fetch_protected_areas(bbox)

    # Pre-compute bboxes for zone features
    dev_zone_bboxes = _precompute_feature_bboxes(dev_zones.get("features", []))
    scheduled_bboxes = _precompute_feature_bboxes(scheduled.get("features", []))
    protected_bboxes = _precompute_feature_bboxes(protected.get("features", []))

    enriched_count = 0

    for building in buildings_geojson.get("features", []):
        b_bbox = _feature_bbox(building)
        props = building.setdefault("properties", {})

        # Default values
        props["in_development_zone"] = False
        props["zone_type"] = None
        props["heritage_scheduled"] = False
        props["protected_area"] = False
        props["protected_area_name"] = None

        if b_bbox is None:
            continue

        # Check development zones
        for feat, f_bbox in dev_zone_bboxes:
            if _bboxes_overlap(b_bbox, f_bbox):
                props["in_development_zone"] = True
                f_props = feat.get("properties", {})
                props["zone_type"] = (
                    f_props.get("zone_type")
                    or f_props.get("ZONE_TYPE")
                    or f_props.get("zone")
                    or f_props.get("ZONE")
                    or f_props.get("category")
                    or f_props.get("CATEGORY")
                )
                break

        # Check scheduled properties
        for feat, f_bbox in scheduled_bboxes:
            if _bboxes_overlap(b_bbox, f_bbox):
                props["heritage_scheduled"] = True
                break

        # Check protected areas
        for feat, f_bbox in protected_bboxes:
            if _bboxes_overlap(b_bbox, f_bbox):
                props["protected_area"] = True
                f_props = feat.get("properties", {})
                props["protected_area_name"] = (
                    f_props.get("site_name")
                    or f_props.get("SITE_NAME")
                    or f_props.get("name")
                    or f_props.get("NAME")
                )
                break

        if props["in_development_zone"] or props["heritage_scheduled"] or props["protected_area"]:
            enriched_count += 1

    logger.info(
        "Enriched %d/%d buildings with planning data",
        enriched_count, len(buildings_geojson.get("features", [])),
    )

    # Add enrichment metadata
    meta = buildings_geojson.setdefault("metadata", {})
    meta["planning_enrichment"] = {
        "enriched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "buildings_enriched": enriched_count,
        "development_zones_count": dev_zones["metadata"]["count"],
        "scheduled_properties_count": scheduled["metadata"]["count"],
        "protected_areas_count": protected["metadata"]["count"],
        "sources": [
            dev_zones["metadata"]["source"],
            scheduled["metadata"]["source"],
            protected["metadata"]["source"],
        ],
    }

    return buildings_geojson


def _precompute_feature_bboxes(
    features: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Tuple[float, float, float, float]]]:
    """
    Compute bounding boxes for a list of features, dropping those without
    valid geometry.
    """
    result = []
    for feat in features:
        fb = _feature_bbox(feat)
        if fb is not None:
            result.append((feat, fb))
    return result


# ---------------------------------------------------------------------------
# CLI entry point: discovery mode
# ---------------------------------------------------------------------------

def _run_discovery() -> None:
    """Print all available layers from MSDI and PA services."""
    print("=" * 72)
    print("MSDI WFS Layers")
    print("=" * 72)

    wfs_layers = discover_layers(MSDI_WFS)
    if wfs_layers:
        for layer in wfs_layers:
            title = layer.get("title", "")
            print(f"  {layer['name']}")
            if title:
                print(f"    Title: {title}")
    else:
        print("  (unavailable or no layers returned)")

    print()
    print("=" * 72)
    print("MSDI WMS Layers")
    print("=" * 72)

    wms_layers = discover_layers(MSDI_WMS)
    if wms_layers:
        for layer in wms_layers:
            title = layer.get("title", "")
            print(f"  {layer['name']}")
            if title:
                print(f"    Title: {title}")
    else:
        print("  (unavailable or no layers returned)")

    print()
    print("=" * 72)
    print("PA MapServer Services")
    print("=" * 72)

    pa_services = discover_pa_mapserver_services()
    if pa_services:
        for svc in pa_services:
            print(f"  {svc['name']} ({svc['type']})")
    else:
        print("  (unavailable or no services returned)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Malta Spatial Data Infrastructure connector",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="List all available WMS/WFS layers and PA MapServer services",
    )
    parser.add_argument(
        "--test-bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Fetch all layers for a test bbox and print summary",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.discover:
        _run_discovery()
        return

    if args.test_bbox:
        bbox = tuple(args.test_bbox)
        print(f"Fetching layers for bbox: {bbox}")
        print()

        for name, fn in [
            ("Development Zones", fetch_development_zones),
            ("Scheduled Properties", fetch_scheduled_properties),
            ("Protected Areas", fetch_protected_areas),
            ("Land Use Zones", fetch_land_use_zones),
            ("Road Centrelines", fetch_road_centrelines),
        ]:
            result = fn(bbox)
            count = result["metadata"]["count"]
            error = result["metadata"].get("error", False)
            status = "ERROR" if error else "OK"
            print(f"  {name:30s} {count:5d} features  [{status}]")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
