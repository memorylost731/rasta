"""
FastAPI router for building geometry and facade classification endpoints.

Mount on the existing Rasta server or run standalone.

Endpoints:
    GET  /api/buildings          - GeoJSON buildings with 3D extrusion
    GET  /api/buildings/facades  - Buildings + facade material classification
    GET  /api/buildings/{osm_id} - Single building detail
    POST /api/buildings/classify - Classify facades from uploaded images
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .osm_buildings import BBox, fetch_buildings
from .building_pipeline import run_pipeline, run_pipeline_osm_only, get_building_by_osm_id
from .facade_classifier import classify_facade

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/buildings", tags=["buildings"])

# In-memory cache for recent pipeline results (keyed by bbox string)
_cache: dict[str, dict] = {}
_cache_facades: dict[str, dict] = {}
CACHE_MAX = 50

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: Optional[str]) -> BBox:
    """Parse bbox query parameter or return Valletta default."""
    if bbox_str:
        try:
            return BBox.from_string(bbox_str)
        except (ValueError, IndexError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid bbox format. Expected 'south,west,north,east': {exc}",
            )
    return BBox.valletta()


def _cache_key(bbox: BBox) -> str:
    return f"{bbox.south:.5f},{bbox.west:.5f},{bbox.north:.5f},{bbox.east:.5f}"


def _put_cache(store: dict, key: str, value: dict) -> None:
    if len(store) >= CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(store))
        del store[oldest]
    store[key] = value


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
async def get_buildings(
    bbox: Optional[str] = Query(
        None,
        description="Bounding box: south,west,north,east (WGS84). Defaults to Valletta, Malta.",
        examples=["35.8940,14.5080,35.9040,14.5200"],
    ),
):
    """
    Fetch building footprints with 3D extrusion properties.

    Returns a GeoJSON FeatureCollection compatible with MapLibre
    fill-extrusion-layer. Each feature includes height, levels,
    material (from OSM), and a rendering color.

    This endpoint does NOT run facade classification -- use
    /api/buildings/facades for material-enriched data.
    """
    parsed_bbox = _parse_bbox(bbox)
    key = _cache_key(parsed_bbox)

    if key in _cache:
        return JSONResponse(content=_cache[key])

    try:
        geojson = await fetch_buildings(parsed_bbox)
    except Exception as exc:
        logger.exception("Failed to fetch buildings from Overpass")
        raise HTTPException(status_code=502, detail=f"Overpass API error: {exc}")

    _put_cache(_cache, key, geojson)
    return JSONResponse(content=geojson)


@router.get("/facades")
async def get_buildings_with_facades(
    bbox: Optional[str] = Query(
        None,
        description="Bounding box: south,west,north,east (WGS84). Defaults to Valletta, Malta.",
    ),
    use_mapillary: bool = Query(
        True,
        description="Use Mapillary street-view images for classification. Set false for OSM-only mode.",
    ),
    mapillary_radius: float = Query(
        30.0,
        description="Search radius (meters) around building centroid for Mapillary images.",
        ge=5, le=200,
    ),
):
    """
    Fetch buildings with facade material classification.

    Runs the full pipeline: OSM footprints + Mapillary imagery +
    Rasta material classification. Each feature is enriched with:
        - facade_material, facade_confidence, facade_method
        - tscm_rf (RF attenuation at sub_1ghz, 2_4ghz, 5ghz)
        - tscm_thermal_conductivity, tscm_thickness_mm
        - color (hex, for MapLibre data-driven styling)

    NOTE: This endpoint is slower than /api/buildings because it
    downloads and classifies street-view images. For large areas,
    set use_mapillary=false for faster results with lower confidence.
    """
    parsed_bbox = _parse_bbox(bbox)
    key = _cache_key(parsed_bbox)

    if not use_mapillary and key in _cache_facades:
        return JSONResponse(content=_cache_facades[key])

    try:
        if use_mapillary:
            result = await run_pipeline(
                bbox=parsed_bbox,
                mapillary_radius_m=mapillary_radius,
            )
        else:
            result = await run_pipeline_osm_only(bbox=parsed_bbox)
    except Exception as exc:
        logger.exception("Building pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    response = result.geojson
    response["metadata"]["elapsed_seconds"] = round(result.elapsed_seconds, 2)
    response["metadata"]["mode"] = result.mode

    if not use_mapillary:
        _put_cache(_cache_facades, key, response)

    return JSONResponse(content=response)


@router.get("/{osm_id}")
async def get_building_detail(
    osm_id: int,
    bbox: Optional[str] = Query(
        None,
        description="Bounding box to search in. Defaults to Valletta.",
    ),
):
    """
    Get detail for a single building by OSM ID.

    Searches the cached or freshly-fetched buildings for the given bbox,
    then runs facade classification on the specific building.
    """
    parsed_bbox = _parse_bbox(bbox)

    try:
        result = await run_pipeline(bbox=parsed_bbox, use_mapillary=True)
    except Exception as exc:
        logger.exception("Pipeline failed for building %s", osm_id)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    feature = get_building_by_osm_id(result.geojson, osm_id)
    if feature is None:
        raise HTTPException(status_code=404, detail=f"Building {osm_id} not found in bbox")

    return JSONResponse(content=feature)


@router.post("/classify")
async def classify_uploaded_facades(
    files: list[UploadFile] = File(..., description="Facade photographs to classify"),
):
    """
    Classify facade materials from uploaded photographs.

    Accepts one or more images. Each is cropped to the facade region
    and classified using Rasta's texture_identify pipeline (Ollama
    vision model first, OpenCV histogram fallback).

    Returns per-image classification with material, confidence,
    RF attenuation, and thermal properties.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per request")

    results = []

    for file in files:
        filename = file.filename or "upload.jpg"
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": filename,
                "error": f"Unsupported file type: {suffix}",
            })
            continue

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            results.append({
                "filename": filename,
                "error": "File too large",
            })
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            classification = classify_facade(str(tmp_path), source_image_id=filename)
            results.append({
                "filename": filename,
                "classification": classification,
            })
        except Exception as exc:
            logger.warning("Classification failed for %s: %s", filename, exc)
            results.append({
                "filename": filename,
                "error": str(exc),
            })
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
# Standalone app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create a standalone FastAPI app with building geometry routes.

        uvicorn rasta.geometry.api:create_app --factory --host 0.0.0.0 --port 8012
    """
    app = FastAPI(
        title="Rasta Building Geometry Engine",
        description=(
            "Building footprint extraction, facade material classification, "
            "and 3D extrusion properties for the PlanO/MapLibre pipeline. "
            "Sources: OpenStreetMap + Mapillary + Rasta material identification."
        ),
        version="1.0.0",
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
            "engine": "rasta-geometry",
            "version": "1.0.0",
        }

    return app
