#!/usr/bin/env python3
"""
Rasta SDK — Quick Start Example

Demonstrates the full material identification and texture pipeline.

Usage:
    pip install rasta[sdk]
    python examples/quickstart.py path/to/wall_photo.jpg
"""

import sys
from pathlib import Path

from rasta.sdk import RastaClient, RastaConnectionError


def main():
    if len(sys.argv) < 2:
        print("Usage: python quickstart.py <photo_path> [server_url]")
        print("  photo_path:  Path to a building material photo")
        print("  server_url:  Rasta server URL (default: http://localhost:8020)")
        sys.exit(1)

    photo = Path(sys.argv[1])
    server = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8020"

    if not photo.exists():
        print(f"File not found: {photo}")
        sys.exit(1)

    with RastaClient(server) as client:
        # 1. Health check
        try:
            health = client.health()
            print(f"Server: {health['engine']} v{health['version']}")
        except RastaConnectionError:
            print(f"Cannot connect to Rasta at {server}")
            sys.exit(1)

        # 2. Full pipeline — identify material + extract textures + scene props
        print(f"\nProcessing: {photo.name}")
        result = client.pipeline(str(photo))

        # 3. Material identification
        mat = result.material
        print(f"\nMaterial: {mat.name} (confidence: {mat.confidence:.0%})")
        print(f"  Method: {mat.method}")
        if mat.model:
            print(f"  Model: {mat.model}")

        # 4. Textures
        tex = result.textures
        print(f"\nTextures (tile: {tex.tile_size_cm}cm):")
        print(f"  Diffuse:   {tex.diffuse}")
        print(f"  Normal:    {tex.normal}")
        print(f"  Roughness: {tex.roughness}")

        # 5. TSCM RF properties
        rf = result.tscm_rf
        if rf:
            print(f"\nTSCM RF Attenuation:")
            print(f"  Material:  {rf.get('material')}")
            print(f"  Thickness: {rf.get('thickness_mm')}mm")
            print(f"  Sub-1GHz:  {rf.get('attenuation_sub_1ghz_db')}dB")
            print(f"  2.4GHz:    {rf.get('attenuation_2_4ghz_db')}dB")
            print(f"  5GHz:      {rf.get('attenuation_5ghz_db')}dB")
            print(f"  RF Class:  {rf.get('rf_class')}")

        # 6. Three.js config
        tj = result.threejs
        if tj:
            print(f"\nThree.js MeshStandardMaterial:")
            print(f"  Roughness: {tj.get('roughness')}")
            print(f"  Metalness: {tj.get('metalness')}")
            print(f"  Color:     {tj.get('color')}")

        # 7. OSM tags
        osm = result.osm_tags
        if osm:
            print(f"\nOSM Tags:")
            for k, v in osm.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
