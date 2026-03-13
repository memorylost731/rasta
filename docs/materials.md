# Materials Database

Rasta supports 23 building materials, each with calibrated physical and RF properties.

## Material Table

| Material | Subcategory | Thickness (mm) | Sub-1GHz (dB) | 2.4GHz (dB) | 5GHz (dB) | Thermal (W/mK) | RF Class |
|----------|-------------|----------------|---------------|-------------|-----------|-----------------|----------|
| concrete | reinforced | 200 | 10.0 | 15.0 | 20.0 | 1.70 | heavy |
| brick | clay_fired | 230 | 5.0 | 8.0 | 12.0 | 0.72 | medium |
| marble | polished | 20 | 8.0 | 12.0 | 16.0 | 2.80 | heavy |
| granite | polished | 30 | 9.0 | 14.0 | 18.0 | 2.50 | heavy |
| limestone | natural | 200 | 7.0 | 10.0 | 14.0 | 1.50 | medium |
| wood_plank | hardwood | 22 | 2.0 | 3.0 | 5.0 | 0.15 | light |
| wood_panel | engineered | 18 | 1.5 | 2.5 | 4.0 | 0.13 | light |
| ceramic_tile | glazed | 10 | 3.0 | 5.0 | 7.0 | 1.00 | medium |
| porcelain_tile | vitrified | 12 | 4.0 | 6.0 | 9.0 | 1.30 | medium |
| glass | float | 6 | 1.0 | 2.5 | 4.0 | 1.00 | light |
| plaster | gypsum | 13 | 1.5 | 3.0 | 5.0 | 0.50 | light |
| stucco | cement | 25 | 3.0 | 5.0 | 8.0 | 0.72 | medium |
| metal_sheet | steel | 2 | 20.0 | 30.0 | 40.0 | 50.00 | opaque |
| metal_panel | aluminum | 4 | 18.0 | 26.0 | 35.0 | 205.00 | opaque |
| stone | natural | 150 | 8.0 | 12.0 | 16.0 | 1.50 | heavy |
| terrazzo | composite | 25 | 7.0 | 10.0 | 14.0 | 1.80 | medium |
| vinyl | sheet | 3 | 0.5 | 1.0 | 1.5 | 0.17 | transparent |
| carpet | synthetic | 10 | 0.3 | 0.5 | 1.0 | 0.06 | transparent |
| linoleum | standard | 4 | 0.5 | 1.0 | 1.5 | 0.17 | transparent |
| cork | natural | 6 | 0.3 | 0.5 | 0.8 | 0.04 | transparent |
| slate | natural | 15 | 6.0 | 9.0 | 13.0 | 2.20 | medium |
| sandstone | natural | 150 | 5.0 | 8.0 | 12.0 | 1.70 | medium |
| render | cement | 20 | 3.0 | 5.0 | 8.0 | 0.90 | medium |

## RF Attenuation Reference

Values calibrated at the listed reference thickness using ITU-R P.2040 and NIST Building Penetration Loss measurements.

### Scaling Formula

```
attenuation_scaled = attenuation_base × (actual_thickness / reference_thickness)
```

Ratio clamped to [0.1, 10.0] to prevent unrealistic values.

### RF Classification Thresholds (at 2.4 GHz)

| Class | Attenuation Range | Description |
|-------|-------------------|-------------|
| transparent | < 2.0 dB | Negligible signal loss |
| light | 2.0 - 5.0 dB | Minor signal degradation |
| medium | 5.0 - 12.0 dB | Noticeable signal loss |
| heavy | 12.0 - 25.0 dB | Significant signal blocking |
| opaque | >= 25.0 dB | Near-total signal blocking |

## OSM Tag Mapping

Each material maps to OpenStreetMap tags:

```
concrete → building:material=concrete, surface=concrete
brick    → building:material=brick, building:facade:material=brick
marble   → building:material=stone, surface=marble
glass    → building:material=glass
metal_*  → building:material=metal
wood_*   → building:material=wood
```

## Adding New Materials

1. Add to `MATERIALS` dict in `rasta/texture_identify.py`
2. Add OpenCV signature to `OPENCV_SIGNATURES` in same file
3. Add Three.js defaults to `THREEJS_DEFAULTS` in `rasta/texture_to_planner.py`
4. Add planner texture to `PLANNER_TEXTURE_MAP` in same file
5. Add tile size to `MATERIAL_TILE_SIZES` in `rasta/texture_extract.py`
6. Update this document
