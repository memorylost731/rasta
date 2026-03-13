"""
Mapillary API v4 client for street-view facade imagery.

Searches for images near a given coordinate, filters by compass angle
to select facade-facing shots, and downloads images for classification.

Requires MAPILLARY_CLIENT_TOKEN environment variable.
Works in degraded mode (no-op) when token is missing.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"
MAPILLARY_TILES_URL = "https://tiles.mapillary.com"

# Search defaults
DEFAULT_RADIUS_M = 50
MAX_IMAGES_PER_QUERY = 20
IMAGE_FIELDS = "id,geometry,compass_angle,captured_at,is_pano,thumb_1024_url,thumb_2048_url,sequence"


@dataclass
class MapillaryImage:
    """A single Mapillary street-view image."""
    image_id: str
    lat: float
    lon: float
    compass_angle: float
    captured_at: int  # Unix timestamp ms
    is_pano: bool
    thumb_1024_url: str
    thumb_2048_url: str
    sequence_id: str
    distance_m: float = 0.0
    angle_to_building: Optional[float] = None


@dataclass
class MapillarySearchResult:
    """Result of a Mapillary image search."""
    images: list[MapillaryImage] = field(default_factory=list)
    query_lat: float = 0.0
    query_lon: float = 0.0
    radius_m: float = 0.0
    token_available: bool = False


def _get_token() -> Optional[str]:
    """Read Mapillary client token from environment."""
    return os.environ.get("MAPILLARY_CLIENT_TOKEN")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two WGS84 points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees (0-360) from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def _angle_diff(a: float, b: float) -> float:
    """Smallest angular difference in degrees (0-180)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _parse_image(raw: dict[str, Any], query_lat: float, query_lon: float) -> Optional[MapillaryImage]:
    """Parse a raw Mapillary API image record."""
    geom = raw.get("geometry", {})
    coords = geom.get("coordinates", [])
    if len(coords) < 2:
        return None

    lon, lat = coords[0], coords[1]

    return MapillaryImage(
        image_id=str(raw.get("id", "")),
        lat=lat,
        lon=lon,
        compass_angle=float(raw.get("compass_angle", 0)),
        captured_at=int(raw.get("captured_at", 0)),
        is_pano=bool(raw.get("is_pano", False)),
        thumb_1024_url=raw.get("thumb_1024_url", ""),
        thumb_2048_url=raw.get("thumb_2048_url", ""),
        sequence_id=raw.get("sequence", ""),
        distance_m=_haversine_m(query_lat, query_lon, lat, lon),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_images(
    lat: float,
    lon: float,
    radius_m: float = DEFAULT_RADIUS_M,
    max_results: int = MAX_IMAGES_PER_QUERY,
    facing_lat: Optional[float] = None,
    facing_lon: Optional[float] = None,
    angle_tolerance: float = 45.0,
) -> MapillarySearchResult:
    """
    Search for Mapillary street-view images near a coordinate.

    Args:
        lat, lon: Query center point (WGS84).
        radius_m: Search radius in meters.
        max_results: Maximum number of images to return.
        facing_lat, facing_lon: If set, filter images whose compass angle
            points roughly toward this coordinate (within angle_tolerance).
        angle_tolerance: Max angular deviation for facade-facing filter (degrees).

    Returns:
        MapillarySearchResult with matching images sorted by distance.
    """
    token = _get_token()
    result = MapillarySearchResult(
        query_lat=lat,
        query_lon=lon,
        radius_m=radius_m,
        token_available=token is not None,
    )

    if token is None:
        logger.warning("MAPILLARY_CLIENT_TOKEN not set, skipping image search")
        return result

    # Mapillary v4: search images by bbox derived from radius
    # Convert radius to approximate degree offset
    lat_offset = radius_m / 111_320
    lon_offset = radius_m / (111_320 * math.cos(math.radians(lat)))
    bbox_str = f"{lon - lon_offset},{lat - lat_offset},{lon + lon_offset},{lat + lat_offset}"

    params = {
        "access_token": token,
        "fields": IMAGE_FIELDS,
        "bbox": bbox_str,
        "limit": min(max_results * 2, 100),  # over-fetch for filtering
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{MAPILLARY_GRAPH_URL}/images", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Mapillary API request failed: %s", exc)
        return result

    raw_images = data.get("data", [])
    images: list[MapillaryImage] = []

    for raw in raw_images:
        img = _parse_image(raw, lat, lon)
        if img is None:
            continue

        # Distance filter (API bbox is approximate)
        if img.distance_m > radius_m:
            continue

        # Facade-facing filter: check if camera points toward the building
        if facing_lat is not None and facing_lon is not None:
            desired_bearing = _bearing_deg(img.lat, img.lon, facing_lat, facing_lon)
            img.angle_to_building = _angle_diff(img.compass_angle, desired_bearing)
            if img.angle_to_building > angle_tolerance:
                continue

        images.append(img)

    # Sort by distance, take top N
    images.sort(key=lambda i: i.distance_m)
    result.images = images[:max_results]

    logger.info(
        "Mapillary search at (%.5f, %.5f) r=%dm: %d raw -> %d filtered",
        lat, lon, int(radius_m), len(raw_images), len(result.images),
    )
    return result


async def download_image(
    image_id: str,
    output_dir: str = "/tmp/mapillary",
    resolution: str = "1024",
) -> Optional[Path]:
    """
    Download a Mapillary image by ID.

    Args:
        image_id: Mapillary image ID.
        output_dir: Directory to save the downloaded image.
        resolution: "1024" or "2048" for thumbnail resolution.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    token = _get_token()
    if token is None:
        logger.warning("MAPILLARY_CLIENT_TOKEN not set, cannot download")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{image_id}.jpg"

    if file_path.exists():
        logger.debug("Image %s already cached at %s", image_id, file_path)
        return file_path

    # Fetch image metadata to get thumbnail URL
    thumb_field = f"thumb_{resolution}_url"
    params = {
        "access_token": token,
        "fields": thumb_field,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Get the thumbnail URL
            resp = await client.get(f"{MAPILLARY_GRAPH_URL}/{image_id}", params=params)
            resp.raise_for_status()
            data = resp.json()

            thumb_url = data.get(thumb_field)
            if not thumb_url:
                logger.warning("No thumbnail URL for image %s", image_id)
                return None

            # Download the actual image
            img_resp = await client.get(thumb_url)
            img_resp.raise_for_status()

            file_path.write_bytes(img_resp.content)
            logger.info("Downloaded image %s (%d bytes) to %s", image_id, len(img_resp.content), file_path)
            return file_path

    except httpx.HTTPError as exc:
        logger.error("Failed to download image %s: %s", image_id, exc)
        return None


async def get_image_metadata(image_id: str) -> Optional[dict[str, Any]]:
    """Fetch full metadata for a single Mapillary image."""
    token = _get_token()
    if token is None:
        return None

    fields = "id,geometry,compass_angle,captured_at,is_pano,thumb_1024_url,thumb_2048_url,sequence,camera_type,height,width"
    params = {"access_token": token, "fields": fields}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{MAPILLARY_GRAPH_URL}/{image_id}", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.error("Failed to get metadata for image %s: %s", image_id, exc)
        return None
