#!/usr/bin/env bash
# Install dependencies for the Rasta building geometry + facade classification pipeline.
# Run on the GPU server (hadrien-skoed-mt) where ~/plano-raster-engine/ lives.
#
# Usage:
#   bash scripts/install-geometry-deps.sh
#   bash scripts/install-geometry-deps.sh --venv /path/to/venv

set -euo pipefail

VENV_PATH="${1:-}"

if [[ "$VENV_PATH" == "--venv" ]]; then
    VENV_PATH="${2:-}"
fi

if [[ -n "$VENV_PATH" ]] && [[ -d "$VENV_PATH" ]]; then
    echo "[*] Activating virtualenv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

echo "[*] Installing geometry pipeline dependencies..."

# Core geospatial
pip install --upgrade \
    httpx \
    overpy \
    shapely \
    geopandas \
    2>&1 | tail -1

# Image processing (headless for server, no X11 needed)
pip install --upgrade \
    opencv-python-headless \
    numpy \
    Pillow \
    2>&1 | tail -1

# 3D mesh (optional, for future trimesh export)
pip install --upgrade \
    trimesh \
    2>&1 | tail -1

# Verify imports
echo ""
echo "[*] Verifying installations..."
python3 -c "
import httpx; print(f'  httpx          {httpx.__version__}')
import overpy; print(f'  overpy         OK')
import shapely; print(f'  shapely        {shapely.__version__}')
import cv2;    print(f'  opencv         {cv2.__version__}')
import numpy;  print(f'  numpy          {numpy.__version__}')
import PIL;    print(f'  Pillow         {PIL.__version__}')
try:
    import geopandas; print(f'  geopandas      {geopandas.__version__}')
except Exception as e:
    print(f'  geopandas      WARN: {e}')
try:
    import trimesh; print(f'  trimesh        {trimesh.__version__}')
except Exception as e:
    print(f'  trimesh        WARN: {e}')
"

echo ""
echo "[+] Geometry pipeline dependencies installed successfully."
echo "    Start the server with:"
echo "    uvicorn rasta.geometry.api:create_app --factory --host 0.0.0.0 --port 8012"
