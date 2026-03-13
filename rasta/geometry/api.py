"""
FastAPI router for building geometry endpoints (lite edition).

Uses buildings_lite + file cache instead of osm_buildings/overpy/shapely.
Only requires `requests` + stdlib on the GPU server.

Mount on the existing Rasta server or run standalone.

Endpoints:
    GET  /api/buildings          - GeoJSON buildings with 3D extrusion + RF properties
    GET  /api/buildings/cities   - List of 19 Malta+Gozo cities with bbox
    GET  /api/buildings/materials - Material properties lookup table
    GET  /api/buildings/facades  - Buildings + facade material classification (heavy)
    GET  /api/buildings/{osm_id} - Single building detail
    POST /api/buildings/classify - Classify facades from uploaded images
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .buildings_lite import fetch_buildings_lite, get_material_properties, VALLETTA_BBOX
from .cache import get_buildings, get_cities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/buildings", tags=["buildings"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Routes — Lite (zero heavy deps)
# ---------------------------------------------------------------------------

@router.get("/cities")
async def list_cities():
    """
    List all 19 Malta + Gozo cities with their bounding boxes.

    Use the bbox values directly as query parameters for /api/buildings.
    """
    cities = get_cities()
    return JSONResponse(content={
        "cities": cities,
        "count": len(cities),
    })


@router.get("/materials")
async def list_materials():
    """
    Return the material-to-properties lookup table.

    Each material includes: color (hex), rf_db (2.4 GHz attenuation in dB),
    rf_class (transparent/light/medium/heavy/opaque), and thermal conductivity.
    """
    return JSONResponse(content=get_material_properties())


@router.get("")
async def get_buildings_endpoint(
    south: Optional[float] = Query(None, description="Southern latitude bound (WGS84)"),
    west: Optional[float] = Query(None, description="Western longitude bound (WGS84)"),
    north: Optional[float] = Query(None, description="Northern latitude bound (WGS84)"),
    east: Optional[float] = Query(None, description="Eastern longitude bound (WGS84)"),
    bbox: Optional[str] = Query(
        None,
        description="Alternative: comma-separated south,west,north,east. Overridden by individual params.",
    ),
):
    """
    Fetch building footprints with 3D extrusion and RF properties.

    Returns a GeoJSON FeatureCollection compatible with MapLibre
    fill-extrusion-layer.  Each feature includes:
        - height, min_height, levels (3D extrusion)
        - material, color (rendering)
        - rf_attenuation_2_4ghz, rf_class (RF analysis)
        - name, addr:street, addr:housenumber, building type
        - osm_id

    If no bbox parameters are provided, defaults to Valletta, Malta.

    Checks file cache (prefetch-cities.py output) first, then falls
    back to live Overpass API query.  Results cached in memory for 1 hour.
    """
    # Resolve bbox from individual params or comma string
    s, w, n, e = _resolve_bbox(south, west, north, east, bbox)

    try:
        geojson = get_buildings(s, w, n, e)
    except Exception as exc:
        logger.exception("Failed to fetch buildings")
        raise HTTPException(status_code=502, detail=f"Building data error: {exc}")

    return JSONResponse(content=geojson)


# ---------------------------------------------------------------------------
# Routes — Heavy (require building_pipeline / facade_classifier)
# These are kept behind lazy imports so the lite endpoints still work
# when the heavy deps are not installed.
# ---------------------------------------------------------------------------

@router.get("/facades")
async def get_buildings_with_facades(
    south: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    bbox: Optional[str] = Query(None),
    use_mapillary: bool = Query(True, description="Use Mapillary images for classification"),
    mapillary_radius: float = Query(30.0, ge=5, le=200),
):
    """
    Fetch buildings with facade material classification (heavy pipeline).

    Requires shapely, httpx, and Ollama.  If those are not available,
    use /api/buildings instead for OSM-only data.
    """
    s, w, n, e = _resolve_bbox(south, west, north, east, bbox)

    try:
        from .osm_buildings import BBox
        from .building_pipeline import run_pipeline, run_pipeline_osm_only
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Heavy pipeline dependencies not available: {exc}. Use /api/buildings for lite mode.",
        )

    parsed_bbox = BBox(s, w, n, e)

    try:
        if use_mapillary:
            result = await run_pipeline(bbox=parsed_bbox, mapillary_radius_m=mapillary_radius)
        else:
            result = await run_pipeline_osm_only(bbox=parsed_bbox)
    except Exception as exc:
        logger.exception("Building pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    response = result.geojson
    response["metadata"]["elapsed_seconds"] = round(result.elapsed_seconds, 2)
    response["metadata"]["mode"] = result.mode

    return JSONResponse(content=response)


@router.get("/{osm_id}")
async def get_building_detail(
    osm_id: str,
    south: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    bbox: Optional[str] = Query(None),
):
    """
    Get detail for a single building by OSM ID.

    Searches the cached/fetched buildings for the given bbox.
    """
    s, w, n, e = _resolve_bbox(south, west, north, east, bbox)

    try:
        geojson = get_buildings(s, w, n, e)
    except Exception as exc:
        logger.exception("Failed to fetch buildings for detail lookup")
        raise HTTPException(status_code=502, detail=str(exc))

    # Search by osm_id (string comparison)
    osm_id_str = str(osm_id)
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if str(props.get("osm_id", "")) == osm_id_str:
            return JSONResponse(content=feature)

    raise HTTPException(status_code=404, detail=f"Building {osm_id} not found in bbox")


@router.post("/classify")
async def classify_uploaded_facades(
    files: list[UploadFile] = File(..., description="Facade photographs to classify"),
):
    """
    Classify facade materials from uploaded photographs.

    Requires the Rasta facade_classifier module (Ollama vision + OpenCV).
    """
    try:
        from .facade_classifier import classify_facade
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Facade classifier not available: {exc}",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per request")

    results = []

    for file in files:
        filename = file.filename or "upload.jpg"
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            results.append({"filename": filename, "error": f"Unsupported file type: {suffix}"})
            continue

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            results.append({"filename": filename, "error": "File too large"})
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            classification = classify_facade(str(tmp_path), source_image_id=filename)
            results.append({"filename": filename, "classification": classification})
        except Exception as exc:
            logger.warning("Classification failed for %s: %s", filename, exc)
            results.append({"filename": filename, "error": str(exc)})
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    return JSONResponse(content={
        "results": results,
        "count": len(results),
        "classified": sum(1 for r in results if "classification" in r),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_bbox(
    south: Optional[float],
    west: Optional[float],
    north: Optional[float],
    east: Optional[float],
    bbox_str: Optional[str],
) -> tuple:
    """
    Resolve bbox from individual query params or comma-separated string.

    Priority: individual params > bbox string > Valletta default.
    """
    # If all four individual params are provided, use them
    if all(v is not None for v in (south, west, north, east)):
        return (south, west, north, east)

    # Try comma-separated string
    if bbox_str:
        try:
            parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(parts) == 4:
                return tuple(parts)
        except (ValueError, IndexError):
            pass
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox format. Expected 'south,west,north,east' with 4 float values.",
        )

    # Default to Valletta
    return VALLETTA_BBOX


# ---------------------------------------------------------------------------
# Standalone app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create a standalone FastAPI app with building geometry routes.

        uvicorn rasta.geometry.api:create_app --factory --host 0.0.0.0 --port 8020
    """
    app = FastAPI(
        title="Rasta Building Geometry Engine (Lite)",
        description=(
            "Building footprint extraction with material classification, "
            "RF attenuation properties, and 3D extrusion data for the "
            "PlanO/MapLibre pipeline.  Source: OpenStreetMap via Overpass API."
        ),
        version="2.0.0",
    )

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://192.168.50.226:5174",
            "http://192.168.50.187:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "engine": "rasta-geometry-lite",
            "version": "2.0.0",
        }

    return app
