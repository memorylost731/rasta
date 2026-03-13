# Rasta — Copilot Instructions

## Project Overview

Rasta is an open-source floor plan recognition and material identification engine.
It replaces RasterScan as the backend for the PlanO floor plan planner.

## Architecture

```
rasta/
├── api.py                    # FastAPI routes (texture pipeline)
├── server.py                 # FastAPI app factory (floor plan + texture)
├── sdk.py                    # Python SDK client for remote Rasta servers
├── texture_identify.py       # Material classification (Ollama vision + OpenCV)
├── texture_extract.py        # PBR texture synthesis (diffuse/normal/roughness)
├── texture_to_planner.py     # react-planner / Three.js / OSM / TSCM mapping
├── floorplan_detect.py       # Floor plan wall/room/door detection (OpenCV)
├── scene_converter_legacy.py # RasterScan JSON → react-planner scene converter
└── __init__.py               # Package metadata
```

## Key Patterns

- **Two-stage classification**: Ollama multimodal vision (primary) → OpenCV histogram (fallback)
- **23 materials**: Each with RF attenuation, thermal conductivity, OSM tags, thickness
- **PBR output**: Diffuse + normal + roughness maps, seamlessly tileable
- **ITU-R P.2040**: RF attenuation calibrated per material at sub-1GHz, 2.4GHz, 5GHz
- **react-planner compatible**: Output scene JSON matches react-planner schema exactly

## Code Style

- Type hints on all public functions
- Docstrings in Google format
- `logging` module (not print)
- OpenCV uses `cv2` with headless build
- NumPy for all array operations
- FastAPI with Pydantic models for request/response validation

## Dependencies

- Python >= 3.10
- fastapi, uvicorn, opencv-python-headless, numpy (core)
- torch, onnxruntime-gpu (optional GPU)
- pdf2image, requests (optional full)
- httpx (SDK client)

## Testing

- pytest with fixtures in `tests/conftest.py`
- Test images in `tests/fixtures/`
- Run: `pytest tests/ -v`

## API Design

- All texture endpoints under `/api/` prefix
- Floor plan endpoints at root (`/upload-plan`, `/analyze`)
- Health check at `/health`
- File uploads via multipart/form-data
- JSON responses everywhere
- CORS configured for PlanO frontend origins

## Material Database

Materials are defined in `texture_identify.py` in the `MATERIALS` dict.
Each entry has: subcategory, thickness_mm, rf_attenuation_db (3 bands),
thermal_conductivity, osm_tags.

## When Adding New Materials

1. Add entry to `MATERIALS` dict in `texture_identify.py`
2. Add OpenCV signature to `OPENCV_SIGNATURES` in same file
3. Add Three.js defaults to `THREEJS_DEFAULTS` in `texture_to_planner.py`
4. Add react-planner texture mapping to `PLANNER_TEXTURE_MAP` in same file
5. Add tile size to `MATERIAL_TILE_SIZES` in `texture_extract.py`
6. Update `docs/materials.md`

## TSCM / RF Properties

RF attenuation follows ITU-R P.2040 + NIST Building Penetration Loss data.
Scaling formula: `attenuation * (actual_thickness / reference_thickness)`
Ratio clamped to [0.1, 10.0].

RF classes: transparent (<2dB), light (2-5), medium (5-12), heavy (12-25), opaque (>=25)
