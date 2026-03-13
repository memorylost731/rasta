# Rasta

Open-source floor plan recognition and material identification engine. Drop-in replacement for RasterScan.

## Features

- **Floor plan detection** — walls, rooms, doors from images/PDFs using OpenCV + Hough lines + morphological analysis
- **Material identification** — classify 23 building materials from photos via CLIP (Ollama) or OpenCV fallback
- **PBR texture synthesis** — generate diffuse, normal, and roughness maps from a single photo
- **React-planner integration** — outputs scene JSON compatible with react-planner
- **Three.js materials** — PBR MeshStandardMaterial configs ready for 3D rendering
- **OSM tagging** — maps materials to OpenStreetMap `building:material`, `surface` tags
- **TSCM RF properties** — calibrated RF attenuation values (ITU-R P.2040) per material at sub-1GHz, 2.4GHz, 5GHz
- **GPU acceleration** — optional CUDA backend with CubiCasa5k segmentation model

## Materials Database

23 materials with calibrated properties:

`concrete` `brick` `marble` `granite` `limestone` `wood_plank` `wood_panel` `ceramic_tile` `porcelain_tile` `glass` `plaster` `stucco` `metal_sheet` `metal_panel` `stone` `terrazzo` `vinyl` `carpet` `linoleum` `cork` `slate` `sandstone` `render`

Each material includes: default thickness, RF attenuation at 3 bands, thermal conductivity, OSM tags.

## Quick Start

```bash
pip install fastapi uvicorn opencv-python-headless numpy
```

### Standalone server

```bash
uvicorn rasta.api:create_app --factory --host 0.0.0.0 --port 8010
```

### Mount on existing FastAPI app

```python
from rasta.api import router as rasta_router
app.include_router(rasta_router)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/identify-material` | Upload photo → material classification |
| POST | `/api/extract-texture` | Upload photo + material → PBR texture maps |
| POST | `/api/texture-pipeline` | Full pipeline: photo → material + textures + scene properties |
| POST | `/api/material-to-scene` | Material → react-planner + Three.js + OSM + TSCM properties |
| GET | `/api/materials` | List all materials with default properties |
| POST | `/upload-plan` | Upload floor plan → react-planner scene JSON |
| POST | `/analyze` | Upload floor plan → raw detection (walls, rooms, doors) |
| GET | `/health` | Health check |

## Architecture

```
Photo → Material ID (CLIP/OpenCV) → Texture Extraction (OpenCV)
  ↓                                        ↓
Material props                    PBR maps (diffuse/normal/roughness)
  ↓                                        ↓
├─ react-planner scene JSON       ├─ Three.js MeshStandardMaterial
├─ OSM tags                       └─ Tile-ready seamless textures
└─ TSCM RF attenuation model
```

## GPU Engine

For production, Rasta supports a GPU-accelerated backend with:
- CubiCasa5k segmentation model (ONNX or PyTorch)
- Celery + Redis for async job queuing
- Multi-worker Gunicorn deployment

## Part of PlanO

Rasta is the recognition engine behind [PlanO](https://github.com/memorylost731/plano_master), an open-source floor plan planner with 2D/3D editing, invoicing, and OSM integration.

## License

MIT
