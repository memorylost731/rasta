# Rasta Python SDK Guide

## Installation

```bash
pip install rasta[sdk]
# or from source
pip install -e ".[sdk]"
```

The SDK requires `httpx` for HTTP communication.

## Quick Start

```python
from rasta.sdk import RastaClient

client = RastaClient("http://localhost:8020")

# Identify material from a photo
result = client.identify_material("wall_photo.jpg")
print(f"{result.name}: {result.confidence:.0%}")

# Full pipeline
pipeline = client.pipeline("brick_wall.jpg")
print(pipeline.material.name)       # "brick"
print(pipeline.tscm_rf)             # RF attenuation data
print(pipeline.threejs)             # Three.js material config
print(pipeline.osm_tags)            # OSM tags

client.close()
```

## Context Manager

```python
with RastaClient("http://gpu-server:8020") as client:
    result = client.pipeline("photo.jpg")
```

## Async Client

```python
import asyncio
from rasta.sdk import AsyncRastaClient

async def main():
    async with AsyncRastaClient("http://gpu-server:8020") as client:
        result = await client.pipeline("photo.jpg")
        print(result.material.name)

asyncio.run(main())
```

## Client Configuration

```python
client = RastaClient(
    base_url="http://gpu-server:8020",
    timeout=120.0,            # Request timeout (seconds)
    verify_ssl=False,         # Skip SSL verification
    headers={"X-Api-Key": "..."}, # Custom headers
)
```

## Result Types

### MaterialResult

```python
result = client.identify_material("photo.jpg")
result.name          # "concrete"
result.confidence    # 0.92
result.subcategory   # "reinforced"
result.method        # "ollama_vision" or "opencv_histogram"
result.model         # "llava:latest" (if Ollama used)
result.properties    # Full properties dict
result.raw           # Original server response
```

### TextureResult

```python
tex = client.extract_texture("photo.jpg", material="marble")
tex.diffuse          # "/textures/marble_diffuse.jpg"
tex.normal           # "/textures/marble_normal.png"
tex.roughness        # "/textures/marble_roughness.png"
tex.tile_size_cm     # 50.0
```

### PipelineResult

```python
result = client.pipeline("photo.jpg", thickness_mm=200)
result.material      # MaterialResult
result.textures      # TextureResult
result.scene         # Full scene dict
result.react_planner # react-planner properties
result.threejs       # Three.js MeshStandardMaterial config
result.osm_tags      # OpenStreetMap tags
result.tscm_rf       # TSCM RF attenuation data
```

### FloorPlanResult

```python
fp = client.analyze_plan("apartment.pdf")
fp.walls             # List of wall segments
fp.rooms             # List of room polygons
fp.doors             # List of door bounding boxes
fp.area              # Total area in px²
fp.perimeter         # Total perimeter in px
```

## Error Handling

```python
from rasta.sdk import (
    RastaClient,
    RastaError,
    RastaConnectionError,
    RastaValidationError,
    RastaServerError,
)

try:
    result = client.pipeline("photo.jpg")
except RastaConnectionError:
    print("Cannot reach server")
except RastaValidationError as e:
    print(f"Invalid input: {e.detail}")
except RastaServerError as e:
    print(f"Server error: {e.status_code}")
except RastaError as e:
    print(f"Rasta error: {e}")
```

## Downloading Textures

```python
result = client.pipeline("wall.jpg")

# Download generated texture files
client.download_texture(result.textures.diffuse, "local_diffuse.jpg")
client.download_texture(result.textures.normal, "local_normal.png")
client.download_texture(result.textures.roughness, "local_roughness.png")
```

## Batch Processing (Async)

```python
import asyncio
from rasta.sdk import AsyncRastaClient

async def process_batch(photos: list[str]):
    async with AsyncRastaClient("http://gpu:8020") as client:
        sem = asyncio.Semaphore(4)  # Limit concurrency

        async def process(path):
            async with sem:
                return await client.pipeline(path)

        return await asyncio.gather(*[process(p) for p in photos])

results = asyncio.run(process_batch(["a.jpg", "b.jpg", "c.jpg"]))
```

## Direct Library Usage (No Server)

You can also use Rasta's core functions directly without a server:

```python
from rasta.texture_identify import identify_material, list_materials
from rasta.texture_extract import extract_texture
from rasta.texture_to_planner import material_to_scene_properties
from rasta.floorplan_detect import analyze_floorplan

# Identify material locally
material = identify_material("photo.jpg")

# Extract textures locally
textures = extract_texture("photo.jpg", material["material"], "/output/")

# Map to scene properties
scene = material_to_scene_properties(material, textures)

# Analyze floor plan
detection = analyze_floorplan("plan.png")
```
