# Rasta — Copilot Instructions

## Project Overview

Rasta is an open-source floor plan recognition and material identification engine.
It replaces RasterScan as the backend for the PlanO floor plan planner.

## Architecture

```
rasta/
├── api.py                    # FastAPI routes (texture pipeline)
├── server.py                 # FastAPI app factory (floor plan + texture + geometry)
├── sdk.py                    # Python SDK client for remote Rasta servers
├── texture_identify.py       # Material classification (Ollama vision + OpenCV)
├── texture_extract.py        # PBR texture synthesis (diffuse/normal/roughness)
├── texture_to_planner.py     # react-planner / Three.js / OSM / TSCM mapping
├── floorplan_detect.py       # Floor plan wall/room/door detection (OpenCV)
├── scene_converter_legacy.py # RasterScan JSON → react-planner scene converter
├── __init__.py               # Package metadata
└── geometry/                 # Building geometry + facade classification pipeline
    ├── __init__.py            # Public API: fetch_buildings, classify_facade, run_pipeline, BBox
    ├── api.py                 # FastAPI router (/api/buildings/*) — GeoJSON + facade endpoints
    ├── osm_buildings.py       # Overpass API client, GeoJSON with 3D extrusion properties
    ├── mapillary_client.py    # Mapillary v4 API — street-view image search + download
    ├── facade_classifier.py   # Per-building facade material classification with aggregation
    └── building_pipeline.py   # Full orchestrator (OSM + Mapillary + Rasta material ID)
```

### Geometry Pipeline

The `rasta.geometry` subpackage adds city-scale building analysis:

1. **OSM Overpass** (`osm_buildings.py`) — Fetches building footprints with height, levels,
   material, and roof tags. Outputs MapLibre fill-extrusion-ready GeoJSON. Default area: Valletta, Malta.
2. **Mapillary** (`mapillary_client.py`) — Searches for street-view images near each building
   centroid, filters by compass angle (facade-facing), downloads thumbnails. Requires
   `MAPILLARY_CLIENT_TOKEN` env var; degrades gracefully when absent.
3. **Facade classifier** (`facade_classifier.py`) — Crops facade region from street-view images,
   runs Rasta's texture_identify pipeline (Ollama vision + OpenCV fallback), aggregates
   per-building classifications with weighted confidence voting.
4. **Pipeline** (`building_pipeline.py`) — Orchestrates all three steps with async batch
   concurrency (semaphore). Enriches each GeoJSON feature with facade_material,
   facade_confidence, TSCM RF attenuation, thermal conductivity, and MapLibre hex colors.
5. **API** (`api.py`) — FastAPI router at `/api/buildings` with in-memory LRU cache.
   Supports full pipeline, OSM-only mode, single-building lookup, and image upload classification.

**Target geography:** Malta + Gozo (19 cities). Pre-fetch script at `scripts/prefetch-cities.py`.

**Key data types:** `BBox`, `PipelineResult`, `BuildingClassification`, `MapillaryImage`, `FacadeCrop`.

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
- httpx (SDK client + geometry pipeline HTTP calls to Overpass/Mapillary)
- torch, onnxruntime-gpu (optional GPU)
- pdf2image, requests (optional full)
- shapely, geopandas (optional geometry extras — see `scripts/install-geometry-deps.sh`)

## Testing

- pytest with fixtures in `tests/conftest.py`
- Test images in `tests/fixtures/`
- Run: `pytest tests/ -v`

## API Design

- All texture endpoints under `/api/` prefix
- Building geometry endpoints under `/api/buildings/` prefix
- Floor plan endpoints at root (`/upload-plan`, `/analyze`)
- Health check at `/health`
- File uploads via multipart/form-data
- JSON responses everywhere (GeoJSON for building endpoints)
- CORS configured for PlanO frontend origins
- Geometry router mounted on main server via `rasta.geometry.api`

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
