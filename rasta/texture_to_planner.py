"""
Maps material classification and texture extraction results to:
    1. React-planner wall/area properties (textureA, textureB, thickness, opacity)
    2. Three.js MeshStandardMaterial configuration
    3. OSM building/surface tags
    4. TSCM RF propagation properties (attenuation at 2.4GHz, 5GHz, sub-1GHz)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# React-planner texture name mapping
# ---------------------------------------------------------------------------

# Map material names to react-planner built-in texture identifiers.
# React-planner typically ships with a limited texture catalog.
# Custom textures are referenced by URL or catalog key.
_PLANNER_TEXTURE_MAP: dict[str, str] = {
    "concrete":       "concrete",
    "brick":          "bricks",
    "marble":         "marble",
    "granite":        "granite",
    "limestone":      "limestone",
    "wood_plank":     "parquet",
    "wood_panel":     "wood",
    "ceramic_tile":   "tiles",
    "porcelain_tile": "tiles",
    "glass":          "glass",
    "plaster":        "plaster",
    "stucco":         "stucco",
    "metal_sheet":    "metallic",
    "metal_panel":    "metallic",
    "stone":          "stone",
    "terrazzo":       "terrazzo",
    "vinyl":          "vinyl",
    "carpet":         "carpet",
    "linoleum":       "linoleum",
    "cork":           "cork",
    "slate":          "slate",
    "sandstone":      "sandstone",
    "render":         "plaster",
}


# ---------------------------------------------------------------------------
# Three.js MeshStandardMaterial defaults per material
# ---------------------------------------------------------------------------

_THREEJS_DEFAULTS: dict[str, dict] = {
    "concrete":       {"roughness": 0.85, "metalness": 0.0,  "color": "#b0b0b0"},
    "brick":          {"roughness": 0.90, "metalness": 0.0,  "color": "#a0522d"},
    "marble":         {"roughness": 0.15, "metalness": 0.0,  "color": "#f0f0f0"},
    "granite":        {"roughness": 0.30, "metalness": 0.05, "color": "#808080"},
    "limestone":      {"roughness": 0.70, "metalness": 0.0,  "color": "#d4c9a8"},
    "wood_plank":     {"roughness": 0.60, "metalness": 0.0,  "color": "#b87333"},
    "wood_panel":     {"roughness": 0.50, "metalness": 0.0,  "color": "#c8a060"},
    "ceramic_tile":   {"roughness": 0.20, "metalness": 0.0,  "color": "#f5f5f5"},
    "porcelain_tile": {"roughness": 0.10, "metalness": 0.05, "color": "#fafafa"},
    "glass":          {"roughness": 0.05, "metalness": 0.1,  "color": "#e0f0ff", "transparent": True, "opacity": 0.3},
    "plaster":        {"roughness": 0.80, "metalness": 0.0,  "color": "#f5f0e8"},
    "stucco":         {"roughness": 0.85, "metalness": 0.0,  "color": "#e8dcc8"},
    "metal_sheet":    {"roughness": 0.25, "metalness": 0.90, "color": "#c0c0c0"},
    "metal_panel":    {"roughness": 0.30, "metalness": 0.85, "color": "#d0d0d0"},
    "stone":          {"roughness": 0.80, "metalness": 0.0,  "color": "#a0a090"},
    "terrazzo":       {"roughness": 0.25, "metalness": 0.0,  "color": "#d0ccc0"},
    "vinyl":          {"roughness": 0.40, "metalness": 0.0,  "color": "#c8c8c0"},
    "carpet":         {"roughness": 0.95, "metalness": 0.0,  "color": "#808070"},
    "linoleum":       {"roughness": 0.45, "metalness": 0.0,  "color": "#b0a890"},
    "cork":           {"roughness": 0.75, "metalness": 0.0,  "color": "#c0a070"},
    "slate":          {"roughness": 0.55, "metalness": 0.0,  "color": "#505060"},
    "sandstone":      {"roughness": 0.70, "metalness": 0.0,  "color": "#d4b896"},
    "render":         {"roughness": 0.75, "metalness": 0.0,  "color": "#e8e0d8"},
}


# ---------------------------------------------------------------------------
# RF attenuation database (dB per single pass through material at default thickness)
# Sources: ITU-R P.2040, NIST Building Penetration Loss measurements
# ---------------------------------------------------------------------------

_RF_ATTENUATION: dict[str, dict[str, float]] = {
    "concrete":       {"sub_1ghz": 10.0, "2_4ghz": 15.0, "5ghz": 20.0},
    "brick":          {"sub_1ghz": 5.0,  "2_4ghz": 8.0,  "5ghz": 12.0},
    "marble":         {"sub_1ghz": 8.0,  "2_4ghz": 12.0, "5ghz": 16.0},
    "granite":        {"sub_1ghz": 9.0,  "2_4ghz": 14.0, "5ghz": 18.0},
    "limestone":      {"sub_1ghz": 7.0,  "2_4ghz": 10.0, "5ghz": 14.0},
    "wood_plank":     {"sub_1ghz": 2.0,  "2_4ghz": 3.0,  "5ghz": 5.0},
    "wood_panel":     {"sub_1ghz": 1.5,  "2_4ghz": 2.5,  "5ghz": 4.0},
    "ceramic_tile":   {"sub_1ghz": 3.0,  "2_4ghz": 5.0,  "5ghz": 7.0},
    "porcelain_tile": {"sub_1ghz": 4.0,  "2_4ghz": 6.0,  "5ghz": 9.0},
    "glass":          {"sub_1ghz": 1.0,  "2_4ghz": 2.5,  "5ghz": 4.0},
    "plaster":        {"sub_1ghz": 1.5,  "2_4ghz": 3.0,  "5ghz": 5.0},
    "stucco":         {"sub_1ghz": 3.0,  "2_4ghz": 5.0,  "5ghz": 8.0},
    "metal_sheet":    {"sub_1ghz": 20.0, "2_4ghz": 30.0, "5ghz": 40.0},
    "metal_panel":    {"sub_1ghz": 18.0, "2_4ghz": 26.0, "5ghz": 35.0},
    "stone":          {"sub_1ghz": 8.0,  "2_4ghz": 12.0, "5ghz": 16.0},
    "terrazzo":       {"sub_1ghz": 7.0,  "2_4ghz": 10.0, "5ghz": 14.0},
    "vinyl":          {"sub_1ghz": 0.5,  "2_4ghz": 1.0,  "5ghz": 1.5},
    "carpet":         {"sub_1ghz": 0.3,  "2_4ghz": 0.5,  "5ghz": 1.0},
    "linoleum":       {"sub_1ghz": 0.5,  "2_4ghz": 1.0,  "5ghz": 1.5},
    "cork":           {"sub_1ghz": 0.3,  "2_4ghz": 0.5,  "5ghz": 0.8},
    "slate":          {"sub_1ghz": 6.0,  "2_4ghz": 9.0,  "5ghz": 13.0},
    "sandstone":      {"sub_1ghz": 5.0,  "2_4ghz": 8.0,  "5ghz": 12.0},
    "render":         {"sub_1ghz": 3.0,  "2_4ghz": 5.0,  "5ghz": 8.0},
}

# Default thicknesses (mm) at which the RF values above are calibrated
_RF_REF_THICKNESS: dict[str, int] = {
    "concrete": 200, "brick": 230, "marble": 20, "granite": 30,
    "limestone": 200, "wood_plank": 22, "wood_panel": 18,
    "ceramic_tile": 10, "porcelain_tile": 12, "glass": 6,
    "plaster": 13, "stucco": 25, "metal_sheet": 2, "metal_panel": 4,
    "stone": 150, "terrazzo": 25, "vinyl": 3, "carpet": 10,
    "linoleum": 4, "cork": 6, "slate": 15, "sandstone": 150,
    "render": 20,
}


# ---------------------------------------------------------------------------
# OSM tag mapping
# ---------------------------------------------------------------------------

_OSM_TAGS: dict[str, dict[str, str]] = {
    "concrete":       {"building:material": "concrete", "surface": "concrete"},
    "brick":          {"building:material": "brick", "building:facade:material": "brick"},
    "marble":         {"building:material": "stone", "surface": "marble"},
    "granite":        {"building:material": "stone", "surface": "granite"},
    "limestone":      {"building:material": "limestone", "surface": "limestone"},
    "wood_plank":     {"building:material": "wood", "surface": "wood"},
    "wood_panel":     {"building:material": "wood", "surface": "wood"},
    "ceramic_tile":   {"surface": "ceramic_tiles", "material": "ceramic"},
    "porcelain_tile": {"surface": "ceramic_tiles", "material": "porcelain"},
    "glass":          {"building:material": "glass", "building:facade:material": "glass"},
    "plaster":        {"building:material": "plaster", "surface": "plaster"},
    "stucco":         {"building:facade:material": "stucco", "surface": "stucco"},
    "metal_sheet":    {"building:material": "metal", "surface": "metal"},
    "metal_panel":    {"building:material": "metal", "building:facade:material": "metal_cladding"},
    "stone":          {"building:material": "stone", "surface": "stone"},
    "terrazzo":       {"surface": "terrazzo", "material": "terrazzo"},
    "vinyl":          {"surface": "vinyl", "material": "vinyl"},
    "carpet":         {"surface": "carpet", "material": "carpet"},
    "linoleum":       {"surface": "linoleum", "material": "linoleum"},
    "cork":           {"surface": "cork", "material": "cork"},
    "slate":          {"surface": "slate", "material": "slate"},
    "sandstone":      {"building:material": "sandstone", "surface": "sandstone"},
    "render":         {"building:facade:material": "render", "surface": "render"},
}


# ---------------------------------------------------------------------------
# Thickness scaling for RF attenuation
# ---------------------------------------------------------------------------

def _scale_rf_attenuation(
    material: str,
    thickness_mm: Optional[int] = None,
) -> dict[str, float]:
    """
    Scale RF attenuation values based on actual wall thickness.

    Uses a log-linear approximation: dB scales roughly linearly with thickness
    for most building materials in the typical range.
    """
    base_rf = _RF_ATTENUATION.get(material)
    if base_rf is None:
        return {"sub_1ghz": 5.0, "2_4ghz": 8.0, "5ghz": 12.0}

    if thickness_mm is None:
        return dict(base_rf)

    ref_thickness = _RF_REF_THICKNESS.get(material, 100)
    if ref_thickness <= 0:
        return dict(base_rf)

    ratio = thickness_mm / ref_thickness
    # Clamp ratio to prevent absurd values
    ratio = max(0.1, min(10.0, ratio))

    return {
        band: round(val * ratio, 1)
        for band, val in base_rf.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def material_to_scene_properties(
    material_result: dict,
    texture_result: dict,
    thickness_override_mm: Optional[int] = None,
    texture_base_url: str = "/textures",
) -> dict:
    """
    Convert material classification + extracted textures into properties for
    react-planner, Three.js, OSM, and TSCM RF analysis.

    Args:
        material_result:      Output from texture_identify.identify_material()
        texture_result:       Output from texture_extract.extract_texture()
        thickness_override_mm: Override material default thickness (mm)
        texture_base_url:     Base URL for serving texture files (default: /textures)

    Returns:
        {
            "react_planner": {
                "textureA": str,
                "textureB": str,
                "thickness": {"length": int},   # in cm for react-planner
                "opacity": float,
                "patternColor": str,
                "custom_diffuse": str,           # URL to custom diffuse texture
            },
            "threejs": {
                "type": "MeshStandardMaterial",
                "map": str,                      # diffuse map URL
                "normalMap": str,                # normal map URL
                "roughnessMap": str,             # roughness map URL
                "roughness": float,
                "metalness": float,
                "color": str,                    # hex fallback color
                "transparent": bool,
                "opacity": float,
            },
            "osm_tags": dict,
            "tscm_rf": {
                "material": str,
                "thickness_mm": int,
                "attenuation_sub_1ghz_db": float,
                "attenuation_2_4ghz_db": float,
                "attenuation_5ghz_db": float,
                "rf_class": str,                 # "transparent", "light", "medium", "heavy", "opaque"
            },
            "tile_size_cm": float,
        }
    """
    material = material_result.get("material", "concrete")
    confidence = material_result.get("confidence", 0.5)
    props = material_result.get("properties", {})

    thickness_mm = thickness_override_mm or props.get("thickness_mm", 200)

    # --- React-planner properties ---
    planner_texture = _PLANNER_TEXTURE_MAP.get(material, "concrete")

    # Build texture URLs from extracted files
    from pathlib import Path
    diffuse_file = Path(texture_result.get("diffuse", "")).name
    normal_file = Path(texture_result.get("normal", "")).name
    roughness_file = Path(texture_result.get("roughness", "")).name

    diffuse_url = f"{texture_base_url}/{diffuse_file}" if diffuse_file else None
    normal_url = f"{texture_base_url}/{normal_file}" if normal_file else None
    roughness_url = f"{texture_base_url}/{roughness_file}" if roughness_file else None

    # Opacity: glass is partially transparent, others are opaque
    is_glass = material == "glass"
    planner_opacity = 0.3 if is_glass else 1.0

    # Thickness in cm for react-planner (which uses cm)
    thickness_cm = max(1, thickness_mm // 10)

    react_planner = {
        "textureA": planner_texture,
        "textureB": planner_texture,
        "thickness": {"length": thickness_cm},
        "opacity": planner_opacity,
        "patternColor": _THREEJS_DEFAULTS.get(material, {}).get("color", "#c0c0c0"),
        "custom_diffuse": diffuse_url,
    }

    # --- Three.js MeshStandardMaterial ---
    threejs_defaults = _THREEJS_DEFAULTS.get(material, {
        "roughness": 0.5, "metalness": 0.0, "color": "#c0c0c0",
    })

    threejs = {
        "type": "MeshStandardMaterial",
        "map": diffuse_url,
        "normalMap": normal_url,
        "roughnessMap": roughness_url,
        "roughness": threejs_defaults["roughness"],
        "metalness": threejs_defaults["metalness"],
        "color": threejs_defaults["color"],
        "transparent": threejs_defaults.get("transparent", False),
        "opacity": threejs_defaults.get("opacity", 1.0),
    }

    # --- OSM tags ---
    osm_tags = dict(_OSM_TAGS.get(material, {"surface": material}))
    osm_tags["wall"] = material
    osm_tags["material"] = material

    # --- TSCM RF properties ---
    rf = _scale_rf_attenuation(material, thickness_mm)

    # Classify RF transparency
    att_24 = rf.get("2_4ghz", 8.0)
    if att_24 < 2.0:
        rf_class = "transparent"
    elif att_24 < 5.0:
        rf_class = "light"
    elif att_24 < 12.0:
        rf_class = "medium"
    elif att_24 < 25.0:
        rf_class = "heavy"
    else:
        rf_class = "opaque"

    tscm_rf = {
        "material": material,
        "thickness_mm": thickness_mm,
        "attenuation_sub_1ghz_db": rf["sub_1ghz"],
        "attenuation_2_4ghz_db": rf["2_4ghz"],
        "attenuation_5ghz_db": rf["5ghz"],
        "rf_class": rf_class,
    }

    return {
        "react_planner": react_planner,
        "threejs": threejs,
        "osm_tags": osm_tags,
        "tscm_rf": tscm_rf,
        "tile_size_cm": texture_result.get("tile_size_cm", 50.0),
        "confidence": confidence,
        "method": material_result.get("method", "unknown"),
    }
