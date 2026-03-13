"""
Rasta Geometry — Building footprint extraction, facade classification, and 3D extrusion.

Extends the Rasta material identification pipeline to work at city scale:
    - OSM Overpass API for building footprints, heights, and material tags
    - Mapillary street-view imagery for facade material classification
    - Pipeline orchestrator that enriches GeoJSON with TSCM RF properties
    - FastAPI endpoints serving MapLibre fill-extrusion-ready data

Modules:
    buildings_lite     - Zero-dep Overpass client (requests only)
    cache              - File-based cache for pre-fetched city data
    malta_gis          - MSDI/PA MapServer connector for planning data
    osm_buildings      - Overpass API client via httpx (heavy)
    mapillary_client   - Mapillary v4 API, image search and download
    facade_classifier  - Per-building facade material classification
    building_pipeline  - Full orchestrator (OSM + Mapillary + Rasta)
    api                - FastAPI router with building/facade endpoints
"""

# Always-available lite imports (only needs requests + stdlib)
from .buildings_lite import fetch_buildings_lite, get_material_properties
from .cache import get_buildings, get_cities
from .malta_gis import (
    fetch_wfs_features,
    fetch_development_zones,
    fetch_scheduled_properties,
    fetch_protected_areas,
    discover_layers,
    enrich_buildings_with_planning,
)

# Heavy imports — optional, fail gracefully when deps missing
try:
    from .osm_buildings import fetch_buildings, BBox
    from .facade_classifier import classify_facade
    from .building_pipeline import run_pipeline
except ImportError:
    fetch_buildings = None
    classify_facade = None
    run_pipeline = None
    BBox = None

__all__ = [
    "fetch_buildings_lite",
    "get_material_properties",
    "get_buildings",
    "get_cities",
    "fetch_buildings",
    "classify_facade",
    "run_pipeline",
    "BBox",
    "fetch_wfs_features",
    "fetch_development_zones",
    "fetch_scheduled_properties",
    "fetch_protected_areas",
    "discover_layers",
    "enrich_buildings_with_planning",
]
