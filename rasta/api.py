"""
FastAPI routes for the Rasta texture and material classification pipeline.

Mount these routes on the existing Rasta FastAPI app or run standalone.

Endpoints:
    POST /api/identify-material     - Upload photo, classify material
    POST /api/extract-texture       - Upload photo + material, generate PBR textures
    POST /api/material-to-scene     - Material + texture data, get scene properties
    POST /api/texture-pipeline      - Full pipeline: photo in, everything out
    GET  /api/materials             - List all known materials with default properties
"""

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .texture_identify import identify_material, list_materials
from .texture_extract import extract_texture
from .texture_to_planner import material_to_scene_properties

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["texture"])

# Texture output directory (served as static files)
TEXTURE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "out" / "textures"
TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> str:
    """Validate uploaded file extension. Returns the suffix."""
    filename = file.filename or "upload.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return suffix


async def _save_upload(file: UploadFile, suffix: str) -> Path:
    """Save uploaded file to a temporary location. Returns the path."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Maximum: {MAX_FILE_SIZE} bytes.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    tmp_path = Path(tmp.name)
    tmp.write(content)
    tmp.close()
    return tmp_path


def _cleanup(path: Path) -> None:
    """Remove a temporary file, ignoring errors."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/identify-material")
async def api_identify_material(file: UploadFile = File(...)):
    """
    Upload a photograph and classify the building material.

    Returns material name, confidence score, physical properties,
    RF attenuation, thermal conductivity, and OSM tags.
    """
    suffix = _validate_upload(file)
    tmp_path = await _save_upload(file, suffix)

    try:
        result = identify_material(str(tmp_path))
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Material identification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")
    finally:
        _cleanup(tmp_path)


@router.post("/extract-texture")
async def api_extract_texture(
    file: UploadFile = File(...),
    material: str = Form("concrete"),
    tile_size: int = Form(512),
    normal_strength: float = Form(2.0),
):
    """
    Upload a photograph and generate PBR texture maps.

    Form fields:
        - file:             The source photograph
        - material:         Material name (for filename and tile size lookup)
        - tile_size:        Output texture resolution in pixels (default 512)
        - normal_strength:  Normal map gradient intensity (default 2.0)

    Returns paths to generated diffuse, normal, and roughness maps.
    """
    suffix = _validate_upload(file)
    tmp_path = await _save_upload(file, suffix)

    try:
        result = extract_texture(
            image_path=str(tmp_path),
            material=material,
            output_dir=str(TEXTURE_OUTPUT_DIR),
            tile_size=tile_size,
            normal_strength=normal_strength,
        )

        # Convert absolute paths to relative URLs
        response = {
            "diffuse": f"/textures/{Path(result['diffuse']).name}",
            "normal": f"/textures/{Path(result['normal']).name}",
            "roughness": f"/textures/{Path(result['roughness']).name}",
            "tile_size_cm": result["tile_size_cm"],
        }
        return JSONResponse(content=response)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Texture extraction failed")
        raise HTTPException(status_code=500, detail=f"Texture extraction failed: {exc}")
    finally:
        _cleanup(tmp_path)


@router.post("/material-to-scene")
async def api_material_to_scene(
    material: str = Form(...),
    confidence: float = Form(0.8),
    subcategory: str = Form(""),
    thickness_mm: int = Form(0),
    diffuse: str = Form(""),
    normal: str = Form(""),
    roughness: str = Form(""),
    tile_size_cm: float = Form(50.0),
    texture_base_url: str = Form("/textures"),
):
    """
    Convert material classification + texture paths into scene properties.

    Returns react-planner properties, Three.js material config,
    OSM tags, and TSCM RF attenuation data.
    """
    # Build the material_result and texture_result dicts expected by the mapper
    material_result = {
        "material": material,
        "confidence": confidence,
        "subcategory": subcategory,
        "properties": {
            "thickness_mm": thickness_mm if thickness_mm > 0 else None,
        },
        "method": "api_input",
    }

    texture_result = {
        "diffuse": diffuse,
        "normal": normal,
        "roughness": roughness,
        "tile_size_cm": tile_size_cm,
    }

    try:
        result = material_to_scene_properties(
            material_result=material_result,
            texture_result=texture_result,
            thickness_override_mm=thickness_mm if thickness_mm > 0 else None,
            texture_base_url=texture_base_url,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Scene property mapping failed")
        raise HTTPException(status_code=500, detail=f"Mapping failed: {exc}")


@router.post("/texture-pipeline")
async def api_texture_pipeline(
    file: UploadFile = File(...),
    tile_size: int = Form(512),
    normal_strength: float = Form(2.0),
    thickness_override_mm: int = Form(0),
    texture_base_url: str = Form("/textures"),
):
    """
    Full texture pipeline: upload a photo and get everything in one call.

    1. Identify material from the photograph
    2. Extract PBR textures (diffuse, normal, roughness)
    3. Map to react-planner, Three.js, OSM, and TSCM properties

    This is the recommended endpoint for end-to-end usage.
    """
    suffix = _validate_upload(file)
    tmp_path = await _save_upload(file, suffix)

    try:
        # Step 1: Identify material
        material_result = identify_material(str(tmp_path))

        material_name = material_result["material"]

        # Step 2: Extract textures
        texture_result = extract_texture(
            image_path=str(tmp_path),
            material=material_name,
            output_dir=str(TEXTURE_OUTPUT_DIR),
            tile_size=tile_size,
            normal_strength=normal_strength,
        )

        # Step 3: Map to scene properties
        thickness = thickness_override_mm if thickness_override_mm > 0 else None
        scene_result = material_to_scene_properties(
            material_result=material_result,
            texture_result=texture_result,
            thickness_override_mm=thickness,
            texture_base_url=texture_base_url,
        )

        # Build unified response
        response = {
            "material": {
                "name": material_name,
                "confidence": material_result["confidence"],
                "subcategory": material_result["subcategory"],
                "method": material_result["method"],
                "model": material_result.get("model"),
            },
            "textures": {
                "diffuse": f"/textures/{Path(texture_result['diffuse']).name}",
                "normal": f"/textures/{Path(texture_result['normal']).name}",
                "roughness": f"/textures/{Path(texture_result['roughness']).name}",
                "tile_size_cm": texture_result["tile_size_cm"],
            },
            "scene": scene_result,
        }
        return JSONResponse(content=response)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Texture pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
    finally:
        _cleanup(tmp_path)


@router.get("/materials")
async def api_list_materials():
    """List all known materials with their default properties."""
    materials = list_materials()
    return JSONResponse(content={"materials": materials, "count": len(materials)})


# ---------------------------------------------------------------------------
# Standalone app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create a standalone FastAPI app with the texture routes.
    Use this if running the texture module independently:

        uvicorn rasta.api:create_app --factory --host 0.0.0.0 --port 8011
    """
    app = FastAPI(
        title="Rasta Texture Engine",
        description=(
            "Material classification and PBR texture synthesis for the PlanO "
            "floor plan recognition pipeline. Identifies building materials from "
            "photographs, generates tileable textures, and maps properties to "
            "react-planner, Three.js, OSM, and TSCM RF models."
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

    # Serve generated textures as static files
    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/textures",
        StaticFiles(directory=str(TEXTURE_OUTPUT_DIR)),
        name="textures",
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "engine": "rasta-texture",
            "version": "1.0.0",
        }

    return app
