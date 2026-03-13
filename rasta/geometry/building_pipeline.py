"""
Full building geometry + facade classification pipeline.

Orchestrates:
    1. Fetch OSM building footprints for a bounding box
    2. Search Mapillary for nearby street-view images per building
    3. Classify facade materials using Rasta's texture_identify
    4. Enrich GeoJSON with material colors, RF properties, TSCM data

Outputs MapLibre-ready GeoJSON with fill-extrusion properties.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .osm_buildings import BBox, fetch_buildings
from .mapillary_client import search_images, download_image, MapillaryImage
from .facade_classifier import (
    classify_building,
    BuildingClassification,
    MATERIAL_DB,
    VALLETTA_DEFAULT_MATERIAL,
)

logger = logging.getLogger(__name__)

# Pipeline configuration defaults
DEFAULT_MAPILLARY_RADIUS_M = 30
MAX_MAPILLARY_PER_BUILDING = 3
BATCH_CONCURRENCY = 5
IMAGE_CACHE_DIR = "/tmp/mapillary"


@dataclass
class PipelineResult:
    """Result of a full building pipeline run."""
    geojson: dict[str, Any]
    building_count: int
    classified_count: int
    mapillary_images_used: int
    elapsed_seconds: float
    bbox: BBox
    mode: str  # "full", "osm_only", "cached"


# ---------------------------------------------------------------------------
# Material-to-color mapping for MapLibre extrusion rendering
# ---------------------------------------------------------------------------

_MATERIAL_COLORS: dict[str, str] = {
    "concrete": "#b0b0b0",
    "brick": "#a0522d",
    "marble": "#f0f0f0",
    "granite": "#808080",
    "limestone": "#d4c9a8",
    "wood_plank": "#b87333",
    "wood_panel": "#c8a060",
    "ceramic_tile": "#f5f5f5",
    "porcelain_tile": "#fafafa",
    "glass": "#e0f0ff",
    "plaster": "#f5f0e8",
    "stucco": "#e8dcc8",
    "metal_sheet": "#c0c0c0",
    "metal_panel": "#d0d0d0",
    "stone": "#a0a090",
    "terrazzo": "#d0ccc0",
    "vinyl": "#c8c8c0",
    "carpet": "#808070",
    "linoleum": "#b0a890",
    "cork": "#c0a070",
    "slate": "#505060",
    "sandstone": "#d4b896",
    "render": "#e8e0d8",
}


def _enrich_feature(
    feature: dict[str, Any],
    classification: Optional[BuildingClassification],
) -> dict[str, Any]:
    """
    Enrich a GeoJSON building feature with facade classification and TSCM data.

    Adds to feature properties:
        - facade_material, facade_confidence, facade_method
        - facade_color (hex, for MapLibre data-driven styling)
        - tscm_rf (RF attenuation at sub_1ghz, 2_4ghz, 5ghz)
        - tscm_thermal_conductivity
    """
    props = feature.get("properties", {})

    if classification:
        props["facade_material"] = classification.material
        props["facade_confidence"] = classification.confidence
        props["facade_method"] = classification.method
        props["facade_subcategory"] = classification.subcategory
        props["facade_crop_count"] = classification.crop_count
        props["color"] = _MATERIAL_COLORS.get(classification.material, "#d4c9a8")

        if classification.rf_attenuation_db:
            props["tscm_rf"] = classification.rf_attenuation_db
        else:
            props["tscm_rf"] = {"sub_1ghz": 7.0, "2_4ghz": 10.0, "5ghz": 14.0}

        db_entry = MATERIAL_DB.get(classification.material, {})
        props["tscm_thermal_conductivity"] = db_entry.get("thermal_conductivity", 1.3)
        props["tscm_thickness_mm"] = db_entry.get("thickness_mm", 200)

    else:
        # No classification available, use defaults
        props["facade_material"] = VALLETTA_DEFAULT_MATERIAL
        props["facade_confidence"] = 0.2
        props["facade_method"] = "none"
        props["color"] = _MATERIAL_COLORS.get(VALLETTA_DEFAULT_MATERIAL, "#d4c9a8")
        props["tscm_rf"] = {"sub_1ghz": 7.0, "2_4ghz": 10.0, "5ghz": 14.0}
        props["tscm_thermal_conductivity"] = 1.3
        props["tscm_thickness_mm"] = 200

    feature["properties"] = props
    return feature


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

async def _classify_single_building(
    feature: dict[str, Any],
    use_mapillary: bool,
    image_cache_dir: str,
    mapillary_radius_m: float,
    max_images: int,
) -> BuildingClassification:
    """Classify facade material for a single building feature."""
    props = feature.get("properties", {})
    osm_id = props.get("osm_id", 0)
    osm_material = props.get("material")
    centroid_lat = props.get("centroid_lat", 0)
    centroid_lon = props.get("centroid_lon", 0)

    image_paths: list[tuple[str, str]] = []

    if use_mapillary and centroid_lat and centroid_lon:
        try:
            result = await search_images(
                lat=centroid_lat,
                lon=centroid_lon,
                radius_m=mapillary_radius_m,
                max_results=max_images,
                facing_lat=centroid_lat,
                facing_lon=centroid_lon,
            )

            if result.images:
                # Download images
                for img in result.images[:max_images]:
                    local_path = await download_image(
                        img.image_id,
                        output_dir=image_cache_dir,
                    )
                    if local_path:
                        image_paths.append((str(local_path), img.image_id))

        except Exception as exc:
            logger.warning("Mapillary search failed for building %s: %s", osm_id, exc)

    return classify_building(
        osm_id=osm_id,
        osm_material=osm_material,
        image_paths=image_paths if image_paths else None,
    )


async def run_pipeline(
    bbox: Optional[BBox] = None,
    use_mapillary: bool = True,
    mapillary_radius_m: float = DEFAULT_MAPILLARY_RADIUS_M,
    max_images_per_building: int = MAX_MAPILLARY_PER_BUILDING,
    batch_size: int = BATCH_CONCURRENCY,
    image_cache_dir: str = IMAGE_CACHE_DIR,
) -> PipelineResult:
    """
    Run the full building geometry + facade classification pipeline.

    Steps:
        1. Fetch OSM buildings for the bounding box
        2. For each building, search Mapillary images near its centroid
        3. Download and classify facade materials
        4. Enrich GeoJSON with material colors and TSCM RF properties

    Args:
        bbox: Bounding box. Defaults to Valletta, Malta.
        use_mapillary: Whether to query Mapillary for street-view images.
            Set False for OSM-only mode (faster, lower confidence).
        mapillary_radius_m: Search radius around building centroid.
        max_images_per_building: Max Mapillary images per building.
        batch_size: Concurrent building classification tasks.
        image_cache_dir: Directory for cached Mapillary downloads.

    Returns:
        PipelineResult with enriched GeoJSON FeatureCollection.
    """
    if bbox is None:
        bbox = BBox.valletta()

    t0 = time.monotonic()

    # Step 1: Fetch OSM buildings
    logger.info("Pipeline: fetching OSM buildings for %s", bbox.to_overpass())
    geojson = await fetch_buildings(bbox)
    features = geojson.get("features", [])
    building_count = len(features)
    logger.info("Pipeline: got %d buildings from OSM", building_count)

    if building_count == 0:
        return PipelineResult(
            geojson=geojson,
            building_count=0,
            classified_count=0,
            mapillary_images_used=0,
            elapsed_seconds=time.monotonic() - t0,
            bbox=bbox,
            mode="osm_only",
        )

    # Step 2+3: Classify facades in batches
    classified_count = 0
    mapillary_images_used = 0
    mode = "full" if use_mapillary else "osm_only"

    semaphore = asyncio.Semaphore(batch_size)

    async def _classify_with_semaphore(feat: dict) -> tuple[dict, Optional[BuildingClassification]]:
        async with semaphore:
            cls = await _classify_single_building(
                feat,
                use_mapillary=use_mapillary,
                image_cache_dir=image_cache_dir,
                mapillary_radius_m=mapillary_radius_m,
                max_images=max_images_per_building,
            )
            return feat, cls

    tasks = [_classify_with_semaphore(f) for f in features]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 4: Enrich features
    enriched_features = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Building classification failed: %s", result)
            continue
        feat, cls = result
        enriched = _enrich_feature(feat, cls)
        enriched_features.append(enriched)
        if cls and cls.method != "none":
            classified_count += 1
            mapillary_images_used += cls.crop_count

    geojson["features"] = enriched_features
    geojson["metadata"]["classified_count"] = classified_count
    geojson["metadata"]["mapillary_images_used"] = mapillary_images_used
    geojson["metadata"]["pipeline_mode"] = mode

    elapsed = time.monotonic() - t0
    logger.info(
        "Pipeline complete: %d buildings, %d classified, %d images, %.1fs",
        building_count, classified_count, mapillary_images_used, elapsed,
    )

    return PipelineResult(
        geojson=geojson,
        building_count=building_count,
        classified_count=classified_count,
        mapillary_images_used=mapillary_images_used,
        elapsed_seconds=elapsed,
        bbox=bbox,
        mode=mode,
    )


async def run_pipeline_osm_only(bbox: Optional[BBox] = None) -> PipelineResult:
    """Convenience: run pipeline without Mapillary (OSM tags + defaults only)."""
    return await run_pipeline(bbox=bbox, use_mapillary=False)


def get_building_by_osm_id(geojson: dict[str, Any], osm_id: int) -> Optional[dict]:
    """Find a single building feature by OSM ID in a GeoJSON FeatureCollection."""
    for feature in geojson.get("features", []):
        if feature.get("properties", {}).get("osm_id") == osm_id:
            return feature
    return None
