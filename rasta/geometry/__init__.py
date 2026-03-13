"""
Rasta Geometry — Building footprint extraction, facade classification, and 3D extrusion.

Extends the Rasta material identification pipeline to work at city scale:
    - OSM Overpass API for building footprints, heights, and material tags
    - Mapillary street-view imagery for facade material classification
    - Pipeline orchestrator that enriches GeoJSON with TSCM RF properties
    - FastAPI endpoints serving MapLibre fill-extrusion-ready data

Modules:
    osm_buildings      - Overpass API client, GeoJSON conversion
    mapillary_client   - Mapillary v4 API, image search and download
    facade_classifier  - Per-building facade material classification
    building_pipeline  - Full orchestrator (OSM + Mapillary + Rasta)
    api                - FastAPI router with building/facade endpoints
"""

from .osm_buildings import fetch_buildings, BBox
from .facade_classifier import classify_facade
from .building_pipeline import run_pipeline

__all__ = [
    "fetch_buildings",
    "classify_facade",
    "run_pipeline",
    "BBox",
]
