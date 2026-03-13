#!/usr/bin/env python3
"""
Rasta SDK — Floor Plan Processing Example

Upload a floor plan image/PDF and get a react-planner scene.

Usage:
    pip install rasta[sdk]
    python examples/floor_plan.py apartment.pdf
"""

import json
import sys
from pathlib import Path

from rasta.sdk import RastaClient


def main():
    if len(sys.argv) < 2:
        print("Usage: python floor_plan.py <plan_image_or_pdf> [server_url]")
        sys.exit(1)

    plan_path = Path(sys.argv[1])
    server = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8020"

    with RastaClient(server) as client:
        # Option A: Get react-planner scene directly
        print("Converting floor plan to react-planner scene...")
        scene = client.upload_plan(str(plan_path))

        layer = scene.get("layers", {}).get("layer-1", {})
        n_vertices = len(layer.get("vertices", {}))
        n_lines = len(layer.get("lines", {}))
        n_areas = len(layer.get("areas", {}))
        print(f"Scene: {n_vertices} vertices, {n_lines} walls, {n_areas} rooms")

        # Save scene JSON
        out = plan_path.stem + "_scene.json"
        Path(out).write_text(json.dumps(scene, indent=2))
        print(f"Saved: {out}")

        # Option B: Get raw detection for custom processing
        print("\nRunning raw analysis...")
        raw = client.analyze_plan(str(plan_path))
        print(f"Detected: {len(raw.walls)} walls, {len(raw.rooms)} rooms, {len(raw.doors)} doors")
        print(f"Total area: {raw.area}px², perimeter: {raw.perimeter:.0f}px")


if __name__ == "__main__":
    main()
