"""
Rasta — Open-Source Floor Plan Recognition & Material Identification Engine
===========================================================================

Part of the PlanO floor plan platform. Drop-in replacement for RasterScan.

Core Modules:
    texture_identify   - Material classification (Ollama vision + OpenCV fallback)
    texture_extract    - PBR texture synthesis (diffuse, normal, roughness maps)
    texture_to_planner - react-planner / Three.js / OSM / TSCM property mapping
    floorplan_detect   - Floor plan wall/room/door detection (OpenCV)
    api                - FastAPI routes for the texture pipeline
    sdk                - Python SDK client for remote Rasta servers

Quick Start (Library):
    from rasta.texture_identify import identify_material
    from rasta.texture_extract import extract_texture
    from rasta.texture_to_planner import material_to_scene_properties

Quick Start (SDK):
    from rasta.sdk import RastaClient

    with RastaClient("http://gpu-server:8020") as client:
        result = client.pipeline("wall_photo.jpg")
        print(result.material.name, result.tscm_rf)
"""

__version__ = "2.0.0"
__author__ = "Open Forged Solutions"
__license__ = "MIT"

__all__ = [
    "texture_identify",
    "texture_extract",
    "texture_to_planner",
    "floorplan_detect",
    "geometry",
    "api",
    "sdk",
]
