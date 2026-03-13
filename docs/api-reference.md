# Rasta API Reference

## Base URL

```
http://your-server:8020
```

## Authentication

None required (designed for internal/LAN deployment).

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "engine": "rasta",
  "version": "2.0.0"
}
```

---

### `POST /upload-plan`

Upload a floor plan image or PDF. Returns a react-planner scene JSON with walls, rooms, and doors.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Floor plan image (PNG/JPG/BMP/TIFF) or PDF |

**Response:** react-planner scene JSON

```json
{
  "unit": "cm",
  "layers": {
    "layer-1": {
      "vertices": { "v_abc123": { "x": 100, "y": 200, "lines": ["l_def456"] } },
      "lines": { "l_def456": { "type": "wall", "vertices": ["v_abc123", "v_ghi789"] } },
      "areas": { "a_jkl012": { "type": "area", "vertices": ["v_abc123", "..."] } }
    }
  },
  "width": 1200,
  "height": 800
}
```

---

### `POST /analyze`

Upload a floor plan and get raw detection results (walls, rooms, doors) without react-planner conversion.

**Request:** Same as `/upload-plan`

**Response:**
```json
{
  "message": "Floor plan analysis successful (open-source engine)",
  "data": {
    "area": 125000,
    "perimeter": 1420.5,
    "walls": [
      { "position": [[100, 200], [500, 200]] }
    ],
    "rooms": [
      [{ "id": "0", "x": 100, "y": 200 }, { "id": "1", "x": 500, "y": 200 }]
    ],
    "doors": [
      { "bbox": [240, 195, 280, 205] }
    ]
  }
}
```

---

### `POST /api/identify-material`

Upload a photograph and classify the building material.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Photo of building surface (JPG/PNG/BMP/TIFF/WebP, max 50MB) |

**Response:**
```json
{
  "material": "brick",
  "confidence": 0.89,
  "subcategory": "clay_fired",
  "method": "ollama_vision",
  "model": "llava:latest",
  "properties": {
    "thickness_mm": 230,
    "rf_attenuation_db": {
      "sub_1ghz": 5.0,
      "2_4ghz": 8.0,
      "5ghz": 12.0
    },
    "thermal_conductivity": 0.72,
    "osm_tags": {
      "building:material": "brick",
      "building:facade:material": "brick"
    }
  }
}
```

**Classification Methods:**
- `ollama_vision` — Multimodal vision model (primary, higher confidence)
- `opencv_histogram` — OpenCV color/texture analysis (fallback, max confidence 0.75)

---

### `POST /api/extract-texture`

Upload a photo and generate PBR texture maps (diffuse, normal, roughness).

**Request:** `multipart/form-data`
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | File | Yes | — | Source photograph |
| `material` | String | No | `concrete` | Material name (for naming and default tile size) |
| `tile_size` | Integer | No | `512` | Output resolution in pixels |
| `normal_strength` | Float | No | `2.0` | Normal map gradient intensity |

**Response:**
```json
{
  "diffuse": "/textures/brick_diffuse.jpg",
  "normal": "/textures/brick_normal.png",
  "roughness": "/textures/brick_roughness.png",
  "tile_size_cm": 40.0
}
```

Texture files are served at the returned paths relative to the server base URL.

---

### `POST /api/material-to-scene`

Convert material classification and texture data into scene properties for react-planner, Three.js, OSM, and TSCM.

**Request:** `multipart/form-data`
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `material` | String | Yes | — | Material name |
| `confidence` | Float | No | `0.8` | Classification confidence |
| `subcategory` | String | No | `""` | Material subcategory |
| `thickness_mm` | Integer | No | `0` | Thickness override (0 = use default) |
| `diffuse` | String | No | `""` | Diffuse texture URL |
| `normal` | String | No | `""` | Normal map URL |
| `roughness` | String | No | `""` | Roughness map URL |
| `tile_size_cm` | Float | No | `50.0` | Tile size in cm |
| `texture_base_url` | String | No | `/textures` | Base URL for texture serving |

**Response:**
```json
{
  "react_planner": {
    "textureA": "bricks",
    "textureB": "bricks",
    "thickness": { "length": 23 },
    "opacity": 1,
    "patternColor": "#c87941"
  },
  "threejs": {
    "type": "MeshStandardMaterial",
    "roughness": 0.75,
    "metalness": 0.0,
    "color": "#c87941",
    "map": "/textures/brick_diffuse.jpg",
    "normalMap": "/textures/brick_normal.png",
    "roughnessMap": "/textures/brick_roughness.png"
  },
  "osm_tags": {
    "building:material": "brick",
    "building:facade:material": "brick",
    "surface": "brick"
  },
  "tscm_rf": {
    "material": "brick",
    "thickness_mm": 230,
    "attenuation_sub_1ghz_db": 5.0,
    "attenuation_2_4ghz_db": 8.0,
    "attenuation_5ghz_db": 12.0,
    "rf_class": "medium"
  }
}
```

---

### `POST /api/texture-pipeline`

**Recommended.** Full pipeline: photo in → material + textures + scene properties out.

Combines identify-material, extract-texture, and material-to-scene in one call.

**Request:** `multipart/form-data`
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | File | Yes | — | Source photograph |
| `tile_size` | Integer | No | `512` | Texture resolution |
| `normal_strength` | Float | No | `2.0` | Normal map intensity |
| `thickness_override_mm` | Integer | No | `0` | Override thickness (0 = default) |
| `texture_base_url` | String | No | `/textures` | Texture URL prefix |

**Response:**
```json
{
  "material": {
    "name": "concrete",
    "confidence": 0.92,
    "subcategory": "reinforced",
    "method": "ollama_vision",
    "model": "llava:latest"
  },
  "textures": {
    "diffuse": "/textures/concrete_diffuse.jpg",
    "normal": "/textures/concrete_normal.png",
    "roughness": "/textures/concrete_roughness.png",
    "tile_size_cm": 50.0
  },
  "scene": {
    "react_planner": { "..." : "..." },
    "threejs": { "..." : "..." },
    "osm_tags": { "..." : "..." },
    "tscm_rf": { "..." : "..." }
  }
}
```

---

### `GET /api/materials`

List all supported materials with default properties.

**Response:**
```json
{
  "materials": [
    {
      "name": "concrete",
      "subcategory": "reinforced",
      "thickness_mm": 200,
      "rf_attenuation_db": { "sub_1ghz": 10.0, "2_4ghz": 15.0, "5ghz": 20.0 },
      "thermal_conductivity": 1.7,
      "osm_tags": { "building:material": "concrete" }
    }
  ],
  "count": 23
}
```

---

## Error Responses

All errors return JSON:
```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (unsupported file type, empty file) |
| 404 | Resource not found |
| 405 | Method not allowed |
| 413 | File too large (>50MB) |
| 422 | Validation error |
| 500 | Internal server error |

---

## CORS

Allowed origins (configurable in server.py):
- `http://localhost:5173-5175`
- `http://192.168.50.226:5174`
- `http://192.168.50.187:5174`

---

## OpenAPI / Swagger

Interactive API docs available at:
- Swagger UI: `http://your-server:8020/docs`
- ReDoc: `http://your-server:8020/redoc`
- OpenAPI JSON: `http://your-server:8020/openapi.json`
