"""
Rasta Python SDK
================

Client library for interacting with a remote Rasta server.

Usage:
    from rasta.sdk import RastaClient

    client = RastaClient("http://localhost:8020")

    # Identify material from a photo
    result = client.identify_material("wall_photo.jpg")
    print(result["material"], result["confidence"])

    # Full pipeline
    pipeline = client.pipeline("brick_wall.jpg", thickness_mm=230)
    print(pipeline["material"]["name"])
    print(pipeline["scene"]["tscm_rf"])

    # List available materials
    materials = client.list_materials()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def _require_httpx() -> None:
    if httpx is None:
        raise ImportError(
            "The Rasta SDK requires httpx. Install it with: pip install httpx"
        )


@dataclass
class RastaConfig:
    """Configuration for the Rasta SDK client."""

    base_url: str = "http://localhost:8020"
    timeout: float = 120.0
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    verify_ssl: bool = True
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MaterialResult:
    """Structured result from material identification."""

    name: str
    confidence: float
    subcategory: str
    method: str
    model: str | None
    properties: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialResult:
        return cls(
            name=data.get("material", "unknown"),
            confidence=data.get("confidence", 0.0),
            subcategory=data.get("subcategory", ""),
            method=data.get("method", ""),
            model=data.get("model"),
            properties=data.get("properties", {}),
            raw=data,
        )


@dataclass
class TextureResult:
    """Structured result from texture extraction."""

    diffuse: str
    normal: str
    roughness: str
    tile_size_cm: float
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextureResult:
        return cls(
            diffuse=data.get("diffuse", ""),
            normal=data.get("normal", ""),
            roughness=data.get("roughness", ""),
            tile_size_cm=data.get("tile_size_cm", 50.0),
            raw=data,
        )


@dataclass
class PipelineResult:
    """Structured result from the full texture pipeline."""

    material: MaterialResult
    textures: TextureResult
    scene: dict[str, Any]
    raw: dict[str, Any]

    @property
    def react_planner(self) -> dict[str, Any]:
        """Get react-planner compatible properties."""
        return self.scene.get("react_planner", {})

    @property
    def threejs(self) -> dict[str, Any]:
        """Get Three.js MeshStandardMaterial config."""
        return self.scene.get("threejs", {})

    @property
    def osm_tags(self) -> dict[str, str]:
        """Get OpenStreetMap tags."""
        return self.scene.get("osm_tags", {})

    @property
    def tscm_rf(self) -> dict[str, Any]:
        """Get TSCM RF attenuation properties."""
        return self.scene.get("tscm_rf", {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        mat_data = data.get("material", {})
        tex_data = data.get("textures", {})
        return cls(
            material=MaterialResult(
                name=mat_data.get("name", "unknown"),
                confidence=mat_data.get("confidence", 0.0),
                subcategory=mat_data.get("subcategory", ""),
                method=mat_data.get("method", ""),
                model=mat_data.get("model"),
                properties={},
                raw=mat_data,
            ),
            textures=TextureResult.from_dict(tex_data),
            scene=data.get("scene", {}),
            raw=data,
        )


@dataclass
class FloorPlanResult:
    """Structured result from floor plan analysis."""

    walls: list[dict[str, Any]]
    rooms: list[list[dict[str, Any]]]
    doors: list[dict[str, Any]]
    area: int
    perimeter: float
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FloorPlanResult:
        d = data.get("data", data)
        return cls(
            walls=d.get("walls", []),
            rooms=d.get("rooms", []),
            doors=d.get("doors", []),
            area=d.get("area", 0),
            perimeter=d.get("perimeter", 0.0),
            raw=data,
        )


class RastaError(Exception):
    """Base exception for Rasta SDK errors."""

    def __init__(self, message: str, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class RastaConnectionError(RastaError):
    """Raised when the client cannot connect to the Rasta server."""
    pass


class RastaValidationError(RastaError):
    """Raised when the server rejects the request (400/422)."""
    pass


class RastaServerError(RastaError):
    """Raised when the server encounters an internal error (500)."""
    pass


class RastaClient:
    """
    Python SDK client for a remote Rasta server.

    Args:
        base_url: Rasta server URL (e.g., "http://localhost:8020")
        timeout: Request timeout in seconds (default 120)
        verify_ssl: Whether to verify SSL certificates (default True)
        headers: Additional headers to send with every request

    Example:
        >>> client = RastaClient("http://gpu-server:8020")
        >>> result = client.identify_material("photo.jpg")
        >>> print(result.name, result.confidence)
        'concrete' 0.92

        >>> pipeline = client.pipeline("wall.jpg")
        >>> print(pipeline.tscm_rf)
        {'material': 'concrete', 'attenuation_2_4ghz_db': 15.0, ...}
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8020",
        timeout: float = 120.0,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        _require_httpx()
        self.config = RastaConfig(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            verify_ssl=verify_ssl,
            headers=headers or {},
        )
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            headers=self.config.headers,
        )

    def __enter__(self) -> RastaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse response and raise appropriate exceptions on error."""
        if response.status_code == 200:
            return response.json()

        detail = None
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        if response.status_code in (400, 413, 422):
            raise RastaValidationError(
                f"Validation error: {detail}",
                status_code=response.status_code,
                detail=detail,
            )
        elif response.status_code >= 500:
            raise RastaServerError(
                f"Server error: {detail}",
                status_code=response.status_code,
                detail=detail,
            )
        else:
            raise RastaError(
                f"HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
                detail=detail,
            )

    def _upload_file(
        self,
        endpoint: str,
        file_path: str | Path,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a file to an endpoint with optional form data."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size > self.config.max_file_size:
            raise RastaValidationError(
                f"File too large: {path.stat().st_size} bytes "
                f"(max {self.config.max_file_size})"
            )

        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                response = self._client.post(endpoint, files=files, data=data or {})
            return self._handle_response(response)
        except httpx.ConnectError as exc:
            raise RastaConnectionError(
                f"Cannot connect to Rasta server at {self.config.base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RastaConnectionError(
                f"Request timed out after {self.config.timeout}s: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Check server health.

        Returns:
            Dict with status, engine name, and version.

        Example:
            >>> client.health()
            {'status': 'ok', 'engine': 'rasta', 'version': '2.0.0'}
        """
        try:
            response = self._client.get("/health")
            return self._handle_response(response)
        except httpx.ConnectError as exc:
            raise RastaConnectionError(
                f"Cannot connect to {self.config.base_url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Material Identification
    # ------------------------------------------------------------------

    def identify_material(self, image_path: str | Path) -> MaterialResult:
        """
        Classify the building material in a photograph.

        Uses Ollama multimodal vision model (primary) with OpenCV
        histogram analysis as fallback.

        Args:
            image_path: Path to photo (JPG, PNG, BMP, TIFF, WebP)

        Returns:
            MaterialResult with name, confidence, properties, and RF data.

        Example:
            >>> result = client.identify_material("brick_wall.jpg")
            >>> print(result.name)
            'brick'
            >>> print(result.confidence)
            0.89
            >>> print(result.properties["rf_attenuation_db"])
            {'sub_1ghz': 5.0, '2_4ghz': 8.0, '5ghz': 12.0}
        """
        data = self._upload_file("/api/identify-material", image_path)
        return MaterialResult.from_dict(data)

    # ------------------------------------------------------------------
    # Texture Extraction
    # ------------------------------------------------------------------

    def extract_texture(
        self,
        image_path: str | Path,
        material: str = "concrete",
        tile_size: int = 512,
        normal_strength: float = 2.0,
    ) -> TextureResult:
        """
        Generate PBR texture maps from a photograph.

        Produces seamlessly tileable diffuse, normal, and roughness maps.

        Args:
            image_path: Path to source photo
            material: Material name (for filename and default tile size)
            tile_size: Output resolution in pixels (default 512)
            normal_strength: Normal map gradient intensity (default 2.0)

        Returns:
            TextureResult with paths to diffuse, normal, roughness maps.

        Example:
            >>> textures = client.extract_texture("marble_floor.jpg", "marble")
            >>> print(textures.diffuse)
            '/textures/marble_diffuse.jpg'
        """
        data = self._upload_file(
            "/api/extract-texture",
            image_path,
            data={
                "material": material,
                "tile_size": str(tile_size),
                "normal_strength": str(normal_strength),
            },
        )
        return TextureResult.from_dict(data)

    # ------------------------------------------------------------------
    # Scene Properties
    # ------------------------------------------------------------------

    def material_to_scene(
        self,
        material: str,
        confidence: float = 0.8,
        subcategory: str = "",
        thickness_mm: int = 0,
        diffuse: str = "",
        normal: str = "",
        roughness: str = "",
        tile_size_cm: float = 50.0,
        texture_base_url: str = "/textures",
    ) -> dict[str, Any]:
        """
        Convert material classification to scene properties.

        Returns properties for react-planner, Three.js, OSM, and TSCM.

        Args:
            material: Material name (e.g., "concrete", "brick")
            confidence: Classification confidence (0-1)
            subcategory: Material subcategory
            thickness_mm: Wall/surface thickness in mm (0 = use default)
            diffuse: Path to diffuse texture map
            normal: Path to normal texture map
            roughness: Path to roughness texture map
            tile_size_cm: Tile size in cm
            texture_base_url: Base URL for texture file serving

        Returns:
            Dict with react_planner, threejs, osm_tags, and tscm_rf sections.

        Example:
            >>> scene = client.material_to_scene("concrete", thickness_mm=200)
            >>> print(scene["tscm_rf"]["attenuation_2_4ghz_db"])
            15.0
            >>> print(scene["osm_tags"]["building:material"])
            'concrete'
        """
        try:
            response = self._client.post(
                "/api/material-to-scene",
                data={
                    "material": material,
                    "confidence": str(confidence),
                    "subcategory": subcategory,
                    "thickness_mm": str(thickness_mm),
                    "diffuse": diffuse,
                    "normal": normal,
                    "roughness": roughness,
                    "tile_size_cm": str(tile_size_cm),
                    "texture_base_url": texture_base_url,
                },
            )
            return self._handle_response(response)
        except httpx.ConnectError as exc:
            raise RastaConnectionError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def pipeline(
        self,
        image_path: str | Path,
        tile_size: int = 512,
        normal_strength: float = 2.0,
        thickness_mm: int = 0,
        texture_base_url: str = "/textures",
    ) -> PipelineResult:
        """
        Full texture pipeline: photo → material + textures + scene properties.

        This is the recommended method for end-to-end usage. Combines
        material identification, texture extraction, and scene property
        mapping in a single server call.

        Args:
            image_path: Path to source photo
            tile_size: Output texture resolution in pixels (default 512)
            normal_strength: Normal map intensity (default 2.0)
            thickness_mm: Override thickness in mm (0 = use default)
            texture_base_url: Base URL for texture serving

        Returns:
            PipelineResult with material, textures, and scene properties.

        Example:
            >>> result = client.pipeline("wall.jpg", thickness_mm=230)
            >>> print(result.material.name)
            'brick'
            >>> print(result.tscm_rf["rf_class"])
            'medium'
            >>> print(result.threejs["roughness"])
            0.75
        """
        data = self._upload_file(
            "/api/texture-pipeline",
            image_path,
            data={
                "tile_size": str(tile_size),
                "normal_strength": str(normal_strength),
                "thickness_override_mm": str(thickness_mm),
                "texture_base_url": texture_base_url,
            },
        )
        return PipelineResult.from_dict(data)

    # ------------------------------------------------------------------
    # Floor Plan
    # ------------------------------------------------------------------

    def upload_plan(self, plan_path: str | Path) -> dict[str, Any]:
        """
        Upload a floor plan and get a react-planner scene JSON.

        Detects walls, rooms, and doors from the floor plan image or PDF,
        then converts the detection into a full react-planner scene.

        Args:
            plan_path: Path to floor plan image (PNG/JPG/BMP/TIFF) or PDF

        Returns:
            react-planner scene JSON with vertices, lines, areas, holes.

        Example:
            >>> scene = client.upload_plan("apartment.pdf")
            >>> print(len(scene["layers"]["layer-1"]["lines"]))
            24
        """
        return self._upload_file("/upload-plan", plan_path)

    def analyze_plan(self, plan_path: str | Path) -> FloorPlanResult:
        """
        Upload a floor plan and get raw detection results.

        Returns detected walls, rooms, and doors without converting
        to react-planner format. Useful for custom processing.

        Args:
            plan_path: Path to floor plan image or PDF

        Returns:
            FloorPlanResult with walls, rooms, doors, area, perimeter.

        Example:
            >>> result = client.analyze_plan("house.png")
            >>> print(f"{len(result.walls)} walls, {len(result.rooms)} rooms")
            '12 walls, 4 rooms'
        """
        data = self._upload_file("/analyze", plan_path)
        return FloorPlanResult.from_dict(data)

    # ------------------------------------------------------------------
    # Materials Database
    # ------------------------------------------------------------------

    def list_materials(self) -> list[dict[str, Any]]:
        """
        List all supported materials with default properties.

        Returns:
            List of material dicts with name, thickness, RF attenuation,
            thermal conductivity, and OSM tags.

        Example:
            >>> materials = client.list_materials()
            >>> for m in materials:
            ...     print(f"{m['name']}: {m['rf_attenuation_db']['2_4ghz']}dB")
            'concrete: 15.0dB'
            'brick: 8.0dB'
            ...
        """
        try:
            response = self._client.get("/api/materials")
            data = self._handle_response(response)
            return data.get("materials", [])
        except httpx.ConnectError as exc:
            raise RastaConnectionError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Texture Download
    # ------------------------------------------------------------------

    def download_texture(
        self,
        texture_path: str,
        output_path: str | Path,
    ) -> Path:
        """
        Download a generated texture file from the server.

        Args:
            texture_path: Server-relative path (e.g., "/textures/concrete_diffuse.jpg")
            output_path: Local path to save the file

        Returns:
            Path to the downloaded file.

        Example:
            >>> result = client.pipeline("wall.jpg")
            >>> client.download_texture(result.textures.diffuse, "diffuse.jpg")
        """
        out = Path(output_path)
        try:
            response = self._client.get(texture_path)
            if response.status_code != 200:
                raise RastaError(
                    f"Failed to download texture: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            out.write_bytes(response.content)
            return out
        except httpx.ConnectError as exc:
            raise RastaConnectionError(str(exc)) from exc


class AsyncRastaClient:
    """
    Async Python SDK client for a remote Rasta server.

    Uses httpx.AsyncClient for non-blocking I/O. Same API as RastaClient.

    Example:
        >>> async with AsyncRastaClient("http://gpu:8020") as client:
        ...     result = await client.pipeline("wall.jpg")
        ...     print(result.material.name)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8020",
        timeout: float = 120.0,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        _require_httpx()
        self.config = RastaConfig(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            verify_ssl=verify_ssl,
            headers=headers or {},
        )
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            headers=self.config.headers,
        )

    async def __aenter__(self) -> AsyncRastaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 200:
            return response.json()
        detail = None
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        if response.status_code in (400, 413, 422):
            raise RastaValidationError(str(detail), response.status_code, detail)
        elif response.status_code >= 500:
            raise RastaServerError(str(detail), response.status_code, detail)
        raise RastaError(f"HTTP {response.status_code}: {detail}", response.status_code, detail)

    async def _upload_file(
        self, endpoint: str, file_path: str | Path, data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                response = await self._client.post(endpoint, files=files, data=data or {})
            return self._handle_response(response)
        except httpx.ConnectError as exc:
            raise RastaConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise RastaConnectionError(f"Timeout: {exc}") from exc

    async def health(self) -> dict[str, Any]:
        """Check server health."""
        response = await self._client.get("/health")
        return self._handle_response(response)

    async def identify_material(self, image_path: str | Path) -> MaterialResult:
        """Classify building material from a photo."""
        data = await self._upload_file("/api/identify-material", image_path)
        return MaterialResult.from_dict(data)

    async def extract_texture(
        self, image_path: str | Path, material: str = "concrete",
        tile_size: int = 512, normal_strength: float = 2.0,
    ) -> TextureResult:
        """Generate PBR textures from a photo."""
        data = await self._upload_file(
            "/api/extract-texture", image_path,
            data={"material": material, "tile_size": str(tile_size), "normal_strength": str(normal_strength)},
        )
        return TextureResult.from_dict(data)

    async def pipeline(
        self, image_path: str | Path, tile_size: int = 512,
        normal_strength: float = 2.0, thickness_mm: int = 0,
        texture_base_url: str = "/textures",
    ) -> PipelineResult:
        """Full pipeline: photo → material + textures + scene."""
        data = await self._upload_file(
            "/api/texture-pipeline", image_path,
            data={
                "tile_size": str(tile_size), "normal_strength": str(normal_strength),
                "thickness_override_mm": str(thickness_mm), "texture_base_url": texture_base_url,
            },
        )
        return PipelineResult.from_dict(data)

    async def upload_plan(self, plan_path: str | Path) -> dict[str, Any]:
        """Upload floor plan → react-planner scene JSON."""
        return await self._upload_file("/upload-plan", plan_path)

    async def analyze_plan(self, plan_path: str | Path) -> FloorPlanResult:
        """Upload floor plan → raw detection (walls, rooms, doors)."""
        data = await self._upload_file("/analyze", plan_path)
        return FloorPlanResult.from_dict(data)

    async def list_materials(self) -> list[dict[str, Any]]:
        """List all supported materials."""
        response = await self._client.get("/api/materials")
        data = self._handle_response(response)
        return data.get("materials", [])

    async def download_texture(self, texture_path: str, output_path: str | Path) -> Path:
        """Download a generated texture file."""
        out = Path(output_path)
        response = await self._client.get(texture_path)
        if response.status_code != 200:
            raise RastaError(f"Download failed: HTTP {response.status_code}", response.status_code)
        out.write_bytes(response.content)
        return out
