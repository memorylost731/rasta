"""
Material identification from photographs.

Primary:   Ollama multimodal model (llava or any vision-capable model at localhost:11434)
Fallback:  OpenCV color histogram + texture descriptor analysis

Returns structured material classification with physical properties,
RF attenuation characteristics, and OSM-compatible tags.
"""

import base64
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Material database: physical + RF + thermal properties
# ---------------------------------------------------------------------------

MATERIAL_DB: dict[str, dict] = {
    "concrete": {
        "subcategory": "reinforced",
        "thickness_mm": 200,
        "rf_attenuation_db": {
            "sub_1ghz": 10.0, "2_4ghz": 15.0, "5ghz": 20.0,
        },
        "thermal_conductivity": 1.7,
        "osm_tags": {"building:material": "concrete", "surface": "concrete"},
    },
    "brick": {
        "subcategory": "clay_fired",
        "thickness_mm": 230,
        "rf_attenuation_db": {
            "sub_1ghz": 5.0, "2_4ghz": 8.0, "5ghz": 12.0,
        },
        "thermal_conductivity": 0.72,
        "osm_tags": {"building:material": "brick", "building:facade:material": "brick"},
    },
    "marble": {
        "subcategory": "polished",
        "thickness_mm": 20,
        "rf_attenuation_db": {
            "sub_1ghz": 8.0, "2_4ghz": 12.0, "5ghz": 16.0,
        },
        "thermal_conductivity": 2.8,
        "osm_tags": {"building:material": "stone", "surface": "marble"},
    },
    "granite": {
        "subcategory": "polished",
        "thickness_mm": 30,
        "rf_attenuation_db": {
            "sub_1ghz": 9.0, "2_4ghz": 14.0, "5ghz": 18.0,
        },
        "thermal_conductivity": 2.9,
        "osm_tags": {"building:material": "stone", "surface": "granite"},
    },
    "limestone": {
        "subcategory": "natural",
        "thickness_mm": 200,
        "rf_attenuation_db": {
            "sub_1ghz": 7.0, "2_4ghz": 10.0, "5ghz": 14.0,
        },
        "thermal_conductivity": 1.3,
        "osm_tags": {"building:material": "limestone", "surface": "limestone"},
    },
    "wood_plank": {
        "subcategory": "hardwood",
        "thickness_mm": 22,
        "rf_attenuation_db": {
            "sub_1ghz": 2.0, "2_4ghz": 3.0, "5ghz": 5.0,
        },
        "thermal_conductivity": 0.15,
        "osm_tags": {"building:material": "wood", "surface": "wood"},
    },
    "wood_panel": {
        "subcategory": "mdf",
        "thickness_mm": 18,
        "rf_attenuation_db": {
            "sub_1ghz": 1.5, "2_4ghz": 2.5, "5ghz": 4.0,
        },
        "thermal_conductivity": 0.14,
        "osm_tags": {"building:material": "wood", "surface": "wood"},
    },
    "ceramic_tile": {
        "subcategory": "glazed",
        "thickness_mm": 10,
        "rf_attenuation_db": {
            "sub_1ghz": 3.0, "2_4ghz": 5.0, "5ghz": 7.0,
        },
        "thermal_conductivity": 1.0,
        "osm_tags": {"surface": "ceramic_tiles", "material": "ceramic"},
    },
    "porcelain_tile": {
        "subcategory": "vitrified",
        "thickness_mm": 12,
        "rf_attenuation_db": {
            "sub_1ghz": 4.0, "2_4ghz": 6.0, "5ghz": 9.0,
        },
        "thermal_conductivity": 1.5,
        "osm_tags": {"surface": "ceramic_tiles", "material": "porcelain"},
    },
    "glass": {
        "subcategory": "float",
        "thickness_mm": 6,
        "rf_attenuation_db": {
            "sub_1ghz": 1.0, "2_4ghz": 2.5, "5ghz": 4.0,
        },
        "thermal_conductivity": 1.0,
        "osm_tags": {"building:material": "glass", "building:facade:material": "glass"},
    },
    "plaster": {
        "subcategory": "gypsum",
        "thickness_mm": 13,
        "rf_attenuation_db": {
            "sub_1ghz": 1.5, "2_4ghz": 3.0, "5ghz": 5.0,
        },
        "thermal_conductivity": 0.5,
        "osm_tags": {"building:material": "plaster", "surface": "plaster"},
    },
    "stucco": {
        "subcategory": "cement_based",
        "thickness_mm": 25,
        "rf_attenuation_db": {
            "sub_1ghz": 3.0, "2_4ghz": 5.0, "5ghz": 8.0,
        },
        "thermal_conductivity": 0.7,
        "osm_tags": {"building:facade:material": "stucco", "surface": "stucco"},
    },
    "metal_sheet": {
        "subcategory": "steel",
        "thickness_mm": 2,
        "rf_attenuation_db": {
            "sub_1ghz": 20.0, "2_4ghz": 30.0, "5ghz": 40.0,
        },
        "thermal_conductivity": 50.0,
        "osm_tags": {"building:material": "metal", "surface": "metal"},
    },
    "metal_panel": {
        "subcategory": "aluminium_composite",
        "thickness_mm": 4,
        "rf_attenuation_db": {
            "sub_1ghz": 18.0, "2_4ghz": 26.0, "5ghz": 35.0,
        },
        "thermal_conductivity": 45.0,
        "osm_tags": {"building:material": "metal", "building:facade:material": "metal_cladding"},
    },
    "stone": {
        "subcategory": "natural_cut",
        "thickness_mm": 150,
        "rf_attenuation_db": {
            "sub_1ghz": 8.0, "2_4ghz": 12.0, "5ghz": 16.0,
        },
        "thermal_conductivity": 2.3,
        "osm_tags": {"building:material": "stone", "surface": "stone"},
    },
    "terrazzo": {
        "subcategory": "polished",
        "thickness_mm": 25,
        "rf_attenuation_db": {
            "sub_1ghz": 7.0, "2_4ghz": 10.0, "5ghz": 14.0,
        },
        "thermal_conductivity": 1.8,
        "osm_tags": {"surface": "terrazzo", "material": "terrazzo"},
    },
    "vinyl": {
        "subcategory": "sheet",
        "thickness_mm": 3,
        "rf_attenuation_db": {
            "sub_1ghz": 0.5, "2_4ghz": 1.0, "5ghz": 1.5,
        },
        "thermal_conductivity": 0.17,
        "osm_tags": {"surface": "vinyl", "material": "vinyl"},
    },
    "carpet": {
        "subcategory": "loop_pile",
        "thickness_mm": 10,
        "rf_attenuation_db": {
            "sub_1ghz": 0.3, "2_4ghz": 0.5, "5ghz": 1.0,
        },
        "thermal_conductivity": 0.06,
        "osm_tags": {"surface": "carpet", "material": "carpet"},
    },
    "linoleum": {
        "subcategory": "natural",
        "thickness_mm": 4,
        "rf_attenuation_db": {
            "sub_1ghz": 0.5, "2_4ghz": 1.0, "5ghz": 1.5,
        },
        "thermal_conductivity": 0.17,
        "osm_tags": {"surface": "linoleum", "material": "linoleum"},
    },
    "cork": {
        "subcategory": "natural",
        "thickness_mm": 6,
        "rf_attenuation_db": {
            "sub_1ghz": 0.3, "2_4ghz": 0.5, "5ghz": 0.8,
        },
        "thermal_conductivity": 0.04,
        "osm_tags": {"surface": "cork", "material": "cork"},
    },
    "slate": {
        "subcategory": "natural",
        "thickness_mm": 15,
        "rf_attenuation_db": {
            "sub_1ghz": 6.0, "2_4ghz": 9.0, "5ghz": 13.0,
        },
        "thermal_conductivity": 2.0,
        "osm_tags": {"surface": "slate", "material": "slate"},
    },
    "sandstone": {
        "subcategory": "natural",
        "thickness_mm": 150,
        "rf_attenuation_db": {
            "sub_1ghz": 5.0, "2_4ghz": 8.0, "5ghz": 12.0,
        },
        "thermal_conductivity": 1.7,
        "osm_tags": {"building:material": "sandstone", "surface": "sandstone"},
    },
    "render": {
        "subcategory": "cement_render",
        "thickness_mm": 20,
        "rf_attenuation_db": {
            "sub_1ghz": 3.0, "2_4ghz": 5.0, "5ghz": 8.0,
        },
        "thermal_conductivity": 0.8,
        "osm_tags": {"building:facade:material": "render", "surface": "render"},
    },
}

ALL_MATERIALS = list(MATERIAL_DB.keys())


# ---------------------------------------------------------------------------
# Ollama multimodal classification
# ---------------------------------------------------------------------------

def _ollama_api(endpoint: str, payload: dict, timeout: int = OLLAMA_TIMEOUT) -> Optional[dict]:
    """POST to Ollama API. Returns parsed JSON or None on failure."""
    url = f"{OLLAMA_BASE}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Ollama API call failed (%s): %s", endpoint, exc)
        return None


def _get_vision_model() -> Optional[str]:
    """Find a vision-capable model on the local Ollama instance."""
    url = f"{OLLAMA_BASE}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    models = data.get("models", [])
    # Prefer known vision models in priority order
    vision_prefixes = ["llava", "bakllava", "moondream", "cogvlm", "minicpm-v"]
    for prefix in vision_prefixes:
        for m in models:
            name = m.get("name", "").lower()
            if prefix in name:
                return m["name"]

    # Check model details for multimodal capability
    for m in models:
        details = m.get("details", {})
        families = details.get("families", [])
        if isinstance(families, list) and any("clip" in f.lower() for f in families):
            return m["name"]

    return None


def _classify_with_ollama(image_path: Path) -> Optional[dict]:
    """Use Ollama vision model for zero-shot material classification."""
    model = _get_vision_model()
    if model is None:
        logger.info("No vision-capable model found on Ollama")
        return None

    logger.info("Using Ollama vision model: %s", model)

    image_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    materials_str = ", ".join(ALL_MATERIALS)
    prompt = (
        "You are a building material classifier. Analyze this photograph and identify "
        "the primary building material or surface material visible.\n\n"
        f"Choose EXACTLY ONE material from this list: {materials_str}\n\n"
        "Respond with ONLY a JSON object in this exact format, nothing else:\n"
        '{"material": "<material_name>", "confidence": <0.0-1.0>, '
        '"subcategory": "<specific_variant>"}\n\n'
        "The subcategory should describe the specific variant, e.g. for brick: "
        '"clay_fired", "engineering", "sandlime"; for wood_plank: "oak", "pine", "walnut".'
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
    }

    result = _ollama_api("/api/generate", payload, timeout=60)
    if result is None:
        return None

    response_text = result.get("response", "").strip()
    logger.debug("Ollama raw response: %s", response_text)

    # Extract JSON from response (model may wrap it in markdown code blocks)
    json_str = response_text
    if "```" in json_str:
        parts = json_str.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                json_str = part
                break

    # Find the JSON object boundaries
    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("Could not extract JSON from Ollama response")
        return None

    try:
        parsed = json.loads(json_str[start:end])
    except json.JSONDecodeError:
        logger.warning("Failed to parse Ollama JSON response")
        return None

    material = parsed.get("material", "").lower().strip()
    if material not in MATERIAL_DB:
        # Try fuzzy match
        for known in ALL_MATERIALS:
            if known in material or material in known:
                material = known
                break
        else:
            logger.warning("Ollama returned unknown material: %s", material)
            return None

    confidence = float(parsed.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    subcategory = parsed.get("subcategory", MATERIAL_DB[material]["subcategory"])

    return {
        "material": material,
        "confidence": confidence,
        "subcategory": str(subcategory),
        "method": "ollama_vision",
        "model": model,
    }


# ---------------------------------------------------------------------------
# OpenCV histogram-based fallback classifier
# ---------------------------------------------------------------------------

# Reference color/texture signatures per material (HSV dominant ranges + texture energy)
# Each entry: (h_range, s_range, v_range, texture_energy_range, edge_density_range)
_CV_SIGNATURES: dict[str, dict] = {
    "concrete": {
        "h": (0, 30), "s": (0, 50), "v": (100, 200),
        "texture_energy": (0.01, 0.08), "edge_density": (0.02, 0.12),
    },
    "brick": {
        "h": (0, 20), "s": (60, 200), "v": (80, 200),
        "texture_energy": (0.05, 0.20), "edge_density": (0.08, 0.25),
    },
    "marble": {
        "h": (0, 180), "s": (0, 40), "v": (180, 255),
        "texture_energy": (0.02, 0.10), "edge_density": (0.03, 0.10),
    },
    "granite": {
        "h": (0, 180), "s": (10, 80), "v": (60, 180),
        "texture_energy": (0.08, 0.25), "edge_density": (0.10, 0.30),
    },
    "limestone": {
        "h": (15, 35), "s": (20, 80), "v": (150, 240),
        "texture_energy": (0.02, 0.08), "edge_density": (0.02, 0.10),
    },
    "wood_plank": {
        "h": (10, 30), "s": (40, 180), "v": (80, 220),
        "texture_energy": (0.05, 0.18), "edge_density": (0.05, 0.18),
    },
    "wood_panel": {
        "h": (10, 30), "s": (30, 150), "v": (100, 230),
        "texture_energy": (0.03, 0.12), "edge_density": (0.03, 0.12),
    },
    "ceramic_tile": {
        "h": (0, 180), "s": (0, 60), "v": (180, 255),
        "texture_energy": (0.01, 0.05), "edge_density": (0.05, 0.20),
    },
    "porcelain_tile": {
        "h": (0, 180), "s": (0, 30), "v": (200, 255),
        "texture_energy": (0.005, 0.03), "edge_density": (0.04, 0.18),
    },
    "glass": {
        "h": (80, 130), "s": (0, 40), "v": (180, 255),
        "texture_energy": (0.001, 0.02), "edge_density": (0.01, 0.05),
    },
    "plaster": {
        "h": (0, 30), "s": (0, 30), "v": (200, 255),
        "texture_energy": (0.005, 0.04), "edge_density": (0.01, 0.06),
    },
    "stucco": {
        "h": (15, 35), "s": (10, 60), "v": (160, 240),
        "texture_energy": (0.04, 0.15), "edge_density": (0.06, 0.18),
    },
    "metal_sheet": {
        "h": (0, 180), "s": (0, 30), "v": (150, 255),
        "texture_energy": (0.001, 0.02), "edge_density": (0.005, 0.04),
    },
    "metal_panel": {
        "h": (0, 180), "s": (0, 40), "v": (140, 250),
        "texture_energy": (0.002, 0.03), "edge_density": (0.01, 0.06),
    },
    "stone": {
        "h": (0, 40), "s": (10, 80), "v": (80, 200),
        "texture_energy": (0.06, 0.22), "edge_density": (0.08, 0.25),
    },
    "terrazzo": {
        "h": (0, 180), "s": (10, 60), "v": (140, 230),
        "texture_energy": (0.06, 0.20), "edge_density": (0.08, 0.22),
    },
    "vinyl": {
        "h": (0, 180), "s": (10, 80), "v": (120, 240),
        "texture_energy": (0.01, 0.05), "edge_density": (0.02, 0.08),
    },
    "carpet": {
        "h": (0, 180), "s": (20, 150), "v": (40, 180),
        "texture_energy": (0.10, 0.35), "edge_density": (0.12, 0.35),
    },
    "linoleum": {
        "h": (0, 180), "s": (10, 80), "v": (100, 220),
        "texture_energy": (0.01, 0.06), "edge_density": (0.02, 0.08),
    },
    "cork": {
        "h": (15, 30), "s": (40, 140), "v": (100, 200),
        "texture_energy": (0.08, 0.25), "edge_density": (0.10, 0.28),
    },
    "slate": {
        "h": (0, 180), "s": (0, 40), "v": (40, 120),
        "texture_energy": (0.04, 0.15), "edge_density": (0.06, 0.20),
    },
    "sandstone": {
        "h": (15, 35), "s": (30, 120), "v": (140, 230),
        "texture_energy": (0.04, 0.15), "edge_density": (0.05, 0.15),
    },
    "render": {
        "h": (0, 40), "s": (0, 40), "v": (180, 255),
        "texture_energy": (0.02, 0.08), "edge_density": (0.02, 0.10),
    },
}


def _compute_image_features(img_bgr: np.ndarray) -> dict:
    """Compute color and texture features from a BGR image."""
    # Resize for consistent analysis
    h, w = img_bgr.shape[:2]
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Dominant HSV values (median of central 60% crop to avoid edges)
    ch, cw = hsv.shape[:2]
    margin_h, margin_w = ch // 5, cw // 5
    center_hsv = hsv[margin_h:ch - margin_h, margin_w:cw - margin_w]
    center_gray = gray[margin_h:ch - margin_h, margin_w:cw - margin_w]

    h_median = float(np.median(center_hsv[:, :, 0]))
    s_median = float(np.median(center_hsv[:, :, 1]))
    v_median = float(np.median(center_hsv[:, :, 2]))

    # Texture energy via Laplacian variance (normalized)
    laplacian = cv2.Laplacian(center_gray, cv2.CV_64F)
    texture_energy = float(np.var(laplacian) / (255.0 ** 2))

    # Edge density via Canny
    edges = cv2.Canny(center_gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Color variance (indicator of uniform vs patterned)
    color_std_h = float(np.std(center_hsv[:, :, 0]))
    color_std_s = float(np.std(center_hsv[:, :, 1]))

    # GLCM-like contrast from co-occurrence approximation
    dx = np.diff(center_gray.astype(np.float64), axis=1)
    dy = np.diff(center_gray.astype(np.float64), axis=0)
    glcm_contrast = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 255.0

    return {
        "h_median": h_median,
        "s_median": s_median,
        "v_median": v_median,
        "texture_energy": texture_energy,
        "edge_density": edge_density,
        "color_std_h": color_std_h,
        "color_std_s": color_std_s,
        "glcm_contrast": glcm_contrast,
    }


def _classify_with_opencv(image_path: Path) -> dict:
    """Classify material using OpenCV color and texture histograms."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    features = _compute_image_features(img)
    scores: dict[str, float] = {}

    for material, sig in _CV_SIGNATURES.items():
        score = 0.0

        # Hue match (circular, 0-180 in OpenCV)
        h_lo, h_hi = sig["h"]
        h_val = features["h_median"]
        if h_lo <= h_val <= h_hi:
            h_range = h_hi - h_lo
            h_center = (h_lo + h_hi) / 2
            h_dist = abs(h_val - h_center) / max(h_range / 2, 1)
            score += (1.0 - h_dist) * 0.15
        else:
            score -= 0.10

        # Saturation match
        s_lo, s_hi = sig["s"]
        s_val = features["s_median"]
        if s_lo <= s_val <= s_hi:
            s_range = s_hi - s_lo
            s_center = (s_lo + s_hi) / 2
            s_dist = abs(s_val - s_center) / max(s_range / 2, 1)
            score += (1.0 - s_dist) * 0.15
        else:
            score -= 0.10

        # Value (brightness) match
        v_lo, v_hi = sig["v"]
        v_val = features["v_median"]
        if v_lo <= v_val <= v_hi:
            v_range = v_hi - v_lo
            v_center = (v_lo + v_hi) / 2
            v_dist = abs(v_val - v_center) / max(v_range / 2, 1)
            score += (1.0 - v_dist) * 0.15
        else:
            score -= 0.10

        # Texture energy match
        te_lo, te_hi = sig["texture_energy"]
        te_val = features["texture_energy"]
        if te_lo <= te_val <= te_hi:
            te_range = te_hi - te_lo
            te_center = (te_lo + te_hi) / 2
            te_dist = abs(te_val - te_center) / max(te_range / 2, 0.001)
            score += (1.0 - min(te_dist, 1.0)) * 0.30
        else:
            # Penalize proportionally to distance from range
            dist_from_range = min(abs(te_val - te_lo), abs(te_val - te_hi))
            score -= min(dist_from_range / max(te_hi, 0.001), 0.3) * 0.30

        # Edge density match
        ed_lo, ed_hi = sig["edge_density"]
        ed_val = features["edge_density"]
        if ed_lo <= ed_val <= ed_hi:
            ed_range = ed_hi - ed_lo
            ed_center = (ed_lo + ed_hi) / 2
            ed_dist = abs(ed_val - ed_center) / max(ed_range / 2, 0.001)
            score += (1.0 - min(ed_dist, 1.0)) * 0.25
        else:
            dist_from_range = min(abs(ed_val - ed_lo), abs(ed_val - ed_hi))
            score -= min(dist_from_range / max(ed_hi, 0.001), 0.3) * 0.25

        scores[material] = score

    # Rank materials by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_material, best_score = ranked[0]

    # Normalize confidence: map raw score range [-1, 1] to [0, 1]
    confidence = max(0.0, min(1.0, (best_score + 0.5) / 1.5))
    # Cap OpenCV confidence lower than Ollama since it is less reliable
    confidence = min(confidence, 0.75)

    return {
        "material": best_material,
        "confidence": round(confidence, 3),
        "subcategory": MATERIAL_DB[best_material]["subcategory"],
        "method": "opencv_histogram",
        "model": None,
        "runner_up": ranked[1][0] if len(ranked) > 1 else None,
        "features": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in features.items()
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_material(image_path: str) -> dict:
    """
    Classify the building/surface material in a photograph.

    Tries Ollama multimodal vision first, falls back to OpenCV histogram analysis.

    Args:
        image_path: Path to the photograph (JPEG, PNG, BMP, TIFF).

    Returns:
        {
            "material": str,          # e.g. "concrete", "brick", "marble"
            "confidence": float,      # 0.0 - 1.0
            "subcategory": str,       # e.g. "reinforced", "clay_fired"
            "properties": {
                "thickness_mm": int,
                "rf_attenuation_db": float,     # at 2.4 GHz
                "thermal_conductivity": float,  # W/(m*K)
            },
            "osm_tags": dict,
            "method": str,            # "ollama_vision" or "opencv_histogram"
            "model": str | None,      # Ollama model name if used
        }

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If image cannot be loaded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {image_path}")

    # Try Ollama vision first
    result = None
    try:
        result = _classify_with_ollama(path)
    except Exception as exc:
        logger.warning("Ollama classification failed: %s", exc)

    # Fall back to OpenCV
    if result is None:
        logger.info("Falling back to OpenCV histogram classifier")
        result = _classify_with_opencv(path)

    material = result["material"]
    db_entry = MATERIAL_DB[material]

    return {
        "material": material,
        "confidence": result["confidence"],
        "subcategory": result["subcategory"],
        "properties": {
            "thickness_mm": db_entry["thickness_mm"],
            "rf_attenuation_db": db_entry["rf_attenuation_db"]["2_4ghz"],
            "thermal_conductivity": db_entry["thermal_conductivity"],
        },
        "osm_tags": db_entry["osm_tags"],
        "method": result["method"],
        "model": result.get("model"),
    }


def get_material_properties(material: str) -> Optional[dict]:
    """Look up properties for a known material name."""
    return MATERIAL_DB.get(material)


def list_materials() -> list[dict]:
    """Return all known materials with their default properties."""
    result = []
    for name, props in MATERIAL_DB.items():
        result.append({
            "material": name,
            "subcategory": props["subcategory"],
            "thickness_mm": props["thickness_mm"],
            "rf_attenuation_db": props["rf_attenuation_db"],
            "thermal_conductivity": props["thermal_conductivity"],
            "osm_tags": props["osm_tags"],
        })
    return result
