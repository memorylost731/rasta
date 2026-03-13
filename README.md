# Rasta

Open-source floor plan recognition and material identification engine. Drop-in replacement for RasterScan.

**Part of [PlanO](https://github.com/memorylost731/plano_master)** — the open-source floor plan platform.

## Features

- **Floor plan detection** — walls, rooms, doors from images/PDFs (OpenCV + Hough lines + morphology)
- **Material identification** — 23 building materials from photos via Ollama vision or OpenCV fallback
- **PBR texture synthesis** — seamless diffuse, normal, and roughness maps from a single photo
- **React-planner integration** — scene JSON compatible with react-planner
- **Three.js materials** — PBR MeshStandardMaterial configs for 3D rendering
- **OSM tagging** — maps materials to OpenStreetMap `building:material` / `surface` tags
- **TSCM RF properties** — calibrated RF attenuation (ITU-R P.2040) at sub-1GHz, 2.4GHz, 5GHz
- **Python SDK** — sync and async clients with typed result objects
- **GPU acceleration** — optional CUDA backend with CubiCasa5k segmentation

## Install

```bash
# Core (server)
pip install rasta

# With SDK client
pip install rasta[sdk]

# Full (server + SDK + PDF support)
pip install rasta[full]

# Development
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Python SDK (recommended)

```python
from rasta.sdk import RastaClient

with RastaClient("http://localhost:8020") as client:
    # Full pipeline: photo → material + textures + scene properties
    result = client.pipeline("brick_wall.jpg")

    print(result.material.name)        # "brick"
    print(result.material.confidence)  # 0.89
    print(result.tscm_rf)             # RF attenuation data
    print(result.threejs)             # Three.js material config
    print(result.osm_tags)            # OSM tags
```

### Option 2: Direct library usage

```python
from rasta.texture_identify import identify_material
from rasta.texture_extract import extract_texture
from rasta.texture_to_planner import material_to_scene_properties

material = identify_material("photo.jpg")
textures = extract_texture("photo.jpg", material["material"], "/output/")
scene = material_to_scene_properties(material, textures)
```

### Option 3: REST API server

```bash
# Standalone
uvicorn rasta.server:app --host 0.0.0.0 --port 8020

# Or mount on existing FastAPI app
from rasta.api import router as rasta_router
app.include_router(rasta_router)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-plan` | Floor plan image/PDF → react-planner scene JSON |
| `POST` | `/analyze` | Floor plan → raw detection (walls, rooms, doors) |
| `POST` | `/api/identify-material` | Photo → material classification |
| `POST` | `/api/extract-texture` | Photo + material → PBR texture maps |
| `POST` | `/api/texture-pipeline` | **Full pipeline**: photo → material + textures + scene |
| `POST` | `/api/material-to-scene` | Material → react-planner + Three.js + OSM + TSCM |
| `GET`  | `/api/materials` | List all 23 materials with properties |
| `GET`  | `/health` | Health check |

Interactive docs at `/docs` (Swagger) and `/redoc`.

## Materials Database

23 materials with calibrated RF attenuation, thermal conductivity, and OSM tags:

| Material | 2.4GHz (dB) | RF Class | Thickness |
|----------|-------------|----------|-----------|
| concrete | 15.0 | heavy | 200mm |
| brick | 8.0 | medium | 230mm |
| glass | 2.5 | light | 6mm |
| metal_sheet | 30.0 | opaque | 2mm |
| wood_plank | 3.0 | light | 22mm |
| carpet | 0.5 | transparent | 10mm |

[Full materials database →](docs/materials.md)

## SDK

The Python SDK provides typed clients for both sync and async usage:

```python
from rasta.sdk import RastaClient, AsyncRastaClient

# Sync
client = RastaClient("http://gpu:8020")
result = client.identify_material("wall.jpg")
print(result.name, result.confidence)

# Async
async with AsyncRastaClient("http://gpu:8020") as client:
    result = await client.pipeline("wall.jpg")
```

**Result types:** `MaterialResult`, `TextureResult`, `PipelineResult`, `FloorPlanResult`
**Exceptions:** `RastaError`, `RastaConnectionError`, `RastaValidationError`, `RastaServerError`

[Full SDK guide →](docs/sdk-guide.md)

## Architecture

```
Photo ──→ Material ID (Ollama/OpenCV) ──→ Texture Extraction (OpenCV)
              │                                    │
              ▼                                    ▼
         Material props                   PBR maps (diffuse/normal/roughness)
              │                                    │
              ├── react-planner scene JSON         ├── Three.js MeshStandardMaterial
              ├── OSM building:material tags       └── Seamless tileable textures
              └── TSCM RF attenuation model

Floor Plan ──→ Wall Detection (Hough) ──→ Room Detection (Contours)
                      │                          │
                      ▼                          ▼
                 Door Detection              Scene Converter
                      │                          │
                      └────────→ react-planner JSON ←──────┘
```

## Documentation

- [API Reference](docs/api-reference.md) — full endpoint documentation
- [SDK Guide](docs/sdk-guide.md) — Python client usage
- [Materials Database](docs/materials.md) — all 23 materials with properties
- [Examples](examples/) — quickstart, async batch, floor plan processing

## Development

```bash
git clone https://github.com/memorylost731/rasta.git
cd rasta
pip install -e ".[dev]"
pytest tests/ -v
ruff check rasta/ tests/
```

## License

MIT — [Open Forged Solutions](https://github.com/memorylost731)
