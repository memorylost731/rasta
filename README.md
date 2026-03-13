# Rasta

Open-source floor plan recognition and material identification engine. Drop-in replacement for RasterScan.

**Part of [PlanO](https://github.com/memorylost731/plano_master)** — the open-source floor plan platform.

## Features

- **Floor plan detection** — walls, rooms, doors from images/PDFs (OpenCV + Hough lines + morphology)
- **Material identification** — 23 building materials from photos via Ollama vision or OpenCV fallback
- **PBR texture synthesis** — seamless diffuse, normal, and roughness maps from a single photo
- **Building geometry pipeline** — city-scale OSM building footprints with 3D extrusion, Mapillary facade classification, and TSCM RF enrichment (Malta + Gozo, 19 cities)
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
| `GET`  | `/api/buildings` | GeoJSON building footprints with 3D extrusion properties |
| `GET`  | `/api/buildings/facades` | Buildings + Mapillary facade material classification |
| `GET`  | `/api/buildings/{osm_id}` | Single building detail with facade classification |
| `POST` | `/api/buildings/classify` | Classify facade materials from uploaded photos |
| `GET`  | `/health` | Health check |

Interactive docs at `/docs` (Swagger) and `/redoc`.

### Building Geometry Endpoints

The `/api/buildings` endpoints serve MapLibre-ready GeoJSON with fill-extrusion properties.

- **`GET /api/buildings?bbox=south,west,north,east`** — Returns building footprints from OSM Overpass. Each feature includes `height`, `levels`, `material`, `color`, and centroid coordinates. Defaults to Valletta, Malta when no bbox is provided.

- **`GET /api/buildings/facades?bbox=...&use_mapillary=true&mapillary_radius=30`** — Full pipeline: fetches OSM footprints, searches Mapillary for nearby street-view images, classifies facade materials using Rasta's texture_identify, and enriches each building with `facade_material`, `facade_confidence`, `tscm_rf` (RF attenuation at 3 bands), and `tscm_thermal_conductivity`. Set `use_mapillary=false` for faster OSM-only mode.

- **`POST /api/buildings/classify`** — Upload up to 20 facade photographs for material classification. Returns per-image material, confidence, and RF properties.

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

## Building Geometry Pipeline

City-scale building footprint extraction with facade material classification, focused on **Malta and Gozo** (19 cities: 13 Malta + 6 Gozo).

```
OSM Overpass API ──→ Building Footprints (GeoJSON)
                           │
                           ├── height, levels, roof_shape, material (from OSM tags)
                           ├── centroid coordinates (for Mapillary image search)
                           │
                     Mapillary v4 API ──→ Street-view images near each building
                           │
                           ├── Compass angle filtering (facade-facing shots)
                           ├── Download + crop facade region
                           │
                     Rasta texture_identify ──→ Material classification
                           │
                           ▼
                     Enriched GeoJSON Feature
                           ├── facade_material, facade_confidence
                           ├── tscm_rf (sub_1ghz, 2_4ghz, 5ghz attenuation dB)
                           ├── tscm_thermal_conductivity, tscm_thickness_mm
                           └── color (hex, MapLibre data-driven styling)
```

**Malta + Gozo cities (19):** Valletta, Sliema, St Julian's, Three Cities, Birkirkara, Mosta, Qormi, Naxxar, Rabat, Mdina, Marsaskala, Mellieha, Swieqi, Victoria (Gozo), Xlendi, Marsalforn, Nadur, Sannat, Gharb.

**Pre-fetch city data:**
```bash
python3 scripts/prefetch-cities.py --all          # All 19 cities
python3 scripts/prefetch-cities.py --city valletta # Single city
python3 scripts/prefetch-cities.py --list          # Show available cities
```

Outputs GeoJSON files + statistics per city to `data/cities/`.

**Environment variables:**
- `MAPILLARY_CLIENT_TOKEN` — required for street-view facade classification. Without it, the pipeline falls back to OSM material tags and Valletta limestone defaults.

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

BBox ──→ OSM Overpass ──→ Building Footprints ──→ Mapillary Search
              │                                        │
              ▼                                        ▼
         GeoJSON with                          Facade Crop + Classify
         3D extrusion props                    (reuses Material ID)
              │                                        │
              └──→ Enriched GeoJSON ←──────────────────┘
                      │
                      ├── MapLibre fill-extrusion layer
                      ├── TSCM RF attenuation per building
                      └── Data-driven color styling
```

## Documentation

- [API Reference](docs/api-reference.md) — full endpoint documentation (including `/api/buildings/*`)
- [SDK Guide](docs/sdk-guide.md) — Python client usage
- [Materials Database](docs/materials.md) — all 23 materials with properties
- [Examples](examples/) — quickstart, async batch, floor plan processing
- [Geometry Setup](scripts/install-geometry-deps.sh) — install geometry pipeline dependencies
- [City Pre-fetcher](scripts/prefetch-cities.py) — download OSM data for Malta + Gozo cities

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
