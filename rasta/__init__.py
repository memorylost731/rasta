"""
Rasta Texture & Material Classification Engine
================================================
Part of the PlanO floor plan recognition pipeline.

Modules:
    texture_identify   - Material classification from photos (Ollama vision + OpenCV fallback)
    texture_extract    - PBR texture synthesis (diffuse, normal, roughness maps)
    texture_to_planner - React-planner / Three.js / OSM / TSCM property mapping
    api                - FastAPI routes for the texture pipeline

Usage:
    from rasta.texture_identify import identify_material
    from rasta.texture_extract import extract_texture
    from rasta.texture_to_planner import material_to_scene_properties
"""

__version__ = "1.0.0"
__all__ = [
    "texture_identify",
    "texture_extract",
    "texture_to_planner",
    "api",
]
