#!/usr/bin/env python3
"""
Rasta SDK — Async Batch Processing Example

Process multiple photos concurrently using the async client.

Usage:
    pip install rasta[sdk]
    python examples/async_batch.py photos_directory/
"""

import asyncio
import sys
from pathlib import Path

from rasta.sdk import AsyncRastaClient


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


async def process_photo(client: AsyncRastaClient, photo: Path) -> dict:
    """Process a single photo through the pipeline."""
    try:
        result = await client.pipeline(str(photo))
        return {
            "file": photo.name,
            "material": result.material.name,
            "confidence": result.material.confidence,
            "rf_class": result.tscm_rf.get("rf_class", "unknown"),
            "attenuation_2_4ghz": result.tscm_rf.get("attenuation_2_4ghz_db", 0),
        }
    except Exception as e:
        return {"file": photo.name, "error": str(e)}


async def main():
    if len(sys.argv) < 2:
        print("Usage: python async_batch.py <photos_dir> [server_url]")
        sys.exit(1)

    photos_dir = Path(sys.argv[1])
    server = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8020"

    photos = [p for p in photos_dir.iterdir() if p.suffix.lower() in SUPPORTED]
    if not photos:
        print(f"No supported images found in {photos_dir}")
        sys.exit(1)

    print(f"Processing {len(photos)} photos from {photos_dir}")
    print(f"Server: {server}\n")

    async with AsyncRastaClient(server) as client:
        # Process up to 4 concurrently
        semaphore = asyncio.Semaphore(4)

        async def bounded(photo: Path):
            async with semaphore:
                return await process_photo(client, photo)

        results = await asyncio.gather(*[bounded(p) for p in photos])

    # Print results table
    print(f"{'File':<30} {'Material':<15} {'Conf':>6} {'RF Class':<12} {'2.4GHz dB':>10}")
    print("-" * 75)
    for r in results:
        if "error" in r:
            print(f"{r['file']:<30} ERROR: {r['error']}")
        else:
            print(
                f"{r['file']:<30} {r['material']:<15} {r['confidence']:>5.0%} "
                f"{r['rf_class']:<12} {r['attenuation_2_4ghz']:>9.1f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
