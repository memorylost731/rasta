"""
Rasta — PlanO Floor Plan Recognition Engine
Open-source replacement for RasterScan. No external API key needed.
Local: OpenCV on-device. Production: GPU engine on hadrien-skoed-mt:8020.

Start: uvicorn server.app:app --host 0.0.0.0 --port 8010
"""

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan_detect import analyze_floorplan
from rasterscan_to_reactplanner import main as convert_to_scene

app = FastAPI(
    title="Rasta — PlanO Floor Plan Recognition",
    description="Open-source floor plan recognition engine. Replaces RasterScan.",
    version="2.0.0",
)

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

OUTDIR = ROOT / "out"
OUTDIR.mkdir(exist_ok=True)


@app.post("/upload-plan")
async def upload_plan(file: UploadFile = File(...)):
    """Upload a floor plan image/PDF → get react-planner scene JSON."""
    suffix = Path(file.filename or "plan").suffix or ".png"
    if suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        # Step 1: Detect walls, rooms, doors
        raw_result = analyze_floorplan(str(tmp_path))

        # Save raw detection for debugging
        raw_json = OUTDIR / "rasterscan_raw.json"
        raw_json.write_text(json.dumps(raw_result, indent=2))

        # Step 2: Convert to react-planner scene format
        scene_json = OUTDIR / "scene.react.json"
        convert_to_scene(str(raw_json), str(scene_json))

        scene = json.loads(scene_json.read_text(encoding="utf-8"))
        return JSONResponse(scene)

    except Exception as e:
        raise HTTPException(500, f"Detection pipeline failed: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/analyze")
async def analyze_only(file: UploadFile = File(...)):
    """Upload a floor plan → get raw detection JSON (walls, rooms, doors)."""
    suffix = Path(file.filename or "plan").suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        result = analyze_floorplan(str(tmp_path))
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


## Mount texture/material pipeline routes
try:
    from rasta.api import router as texture_router
    app.include_router(texture_router)
except ImportError:
    pass

## Mount building geometry + facade classification routes
try:
    from rasta.geometry.api import router as geometry_router
    app.include_router(geometry_router)
except ImportError:
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "rasta", "version": "2.0.0"}
