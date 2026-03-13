"""Tests for the Rasta Python SDK."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rasta.sdk import (
    AsyncRastaClient,
    FloorPlanResult,
    MaterialResult,
    PipelineResult,
    RastaClient,
    RastaConfig,
    RastaConnectionError,
    RastaError,
    RastaServerError,
    RastaValidationError,
    TextureResult,
)


class TestRastaConfig:
    def test_defaults(self):
        config = RastaConfig()
        assert config.base_url == "http://localhost:8020"
        assert config.timeout == 120.0
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.verify_ssl is True

    def test_custom(self):
        config = RastaConfig(base_url="http://gpu:8020", timeout=30.0)
        assert config.base_url == "http://gpu:8020"
        assert config.timeout == 30.0


class TestMaterialResult:
    def test_from_dict(self):
        data = {
            "material": "brick",
            "confidence": 0.89,
            "subcategory": "clay_fired",
            "method": "ollama_vision",
            "model": "llava:latest",
            "properties": {"thickness_mm": 230},
        }
        result = MaterialResult.from_dict(data)
        assert result.name == "brick"
        assert result.confidence == 0.89
        assert result.subcategory == "clay_fired"
        assert result.method == "ollama_vision"
        assert result.model == "llava:latest"

    def test_from_dict_missing_fields(self):
        result = MaterialResult.from_dict({})
        assert result.name == "unknown"
        assert result.confidence == 0.0


class TestTextureResult:
    def test_from_dict(self):
        data = {
            "diffuse": "/textures/brick_diffuse.jpg",
            "normal": "/textures/brick_normal.png",
            "roughness": "/textures/brick_roughness.png",
            "tile_size_cm": 40.0,
        }
        result = TextureResult.from_dict(data)
        assert result.diffuse == "/textures/brick_diffuse.jpg"
        assert result.tile_size_cm == 40.0


class TestPipelineResult:
    def test_from_dict(self):
        data = {
            "material": {"name": "concrete", "confidence": 0.9, "method": "ollama_vision"},
            "textures": {
                "diffuse": "/textures/d.jpg",
                "normal": "/textures/n.png",
                "roughness": "/textures/r.png",
                "tile_size_cm": 50.0,
            },
            "scene": {
                "react_planner": {"textureA": "concrete"},
                "threejs": {"roughness": 0.85},
                "osm_tags": {"building:material": "concrete"},
                "tscm_rf": {"rf_class": "heavy", "attenuation_2_4ghz_db": 15.0},
            },
        }
        result = PipelineResult.from_dict(data)
        assert result.material.name == "concrete"
        assert result.textures.tile_size_cm == 50.0
        assert result.react_planner["textureA"] == "concrete"
        assert result.threejs["roughness"] == 0.85
        assert result.osm_tags["building:material"] == "concrete"
        assert result.tscm_rf["rf_class"] == "heavy"


class TestFloorPlanResult:
    def test_from_dict(self):
        data = {
            "data": {
                "walls": [{"position": [[0, 0], [100, 0]]}],
                "rooms": [[{"id": "0", "x": 0, "y": 0}]],
                "doors": [{"bbox": [40, 0, 60, 10]}],
                "area": 10000,
                "perimeter": 400.0,
            }
        }
        result = FloorPlanResult.from_dict(data)
        assert len(result.walls) == 1
        assert len(result.rooms) == 1
        assert len(result.doors) == 1
        assert result.area == 10000


class TestRastaErrors:
    def test_error_hierarchy(self):
        assert issubclass(RastaConnectionError, RastaError)
        assert issubclass(RastaValidationError, RastaError)
        assert issubclass(RastaServerError, RastaError)

    def test_error_attributes(self):
        err = RastaError("test", status_code=400, detail="bad input")
        assert str(err) == "test"
        assert err.status_code == 400
        assert err.detail == "bad input"


class TestRastaClientInit:
    def test_strips_trailing_slash(self):
        with patch("rasta.sdk.httpx") as mock_httpx:
            mock_httpx.Client = MagicMock()
            client = RastaClient("http://server:8020/")
            assert client.config.base_url == "http://server:8020"

    def test_context_manager(self):
        with patch("rasta.sdk.httpx") as mock_httpx:
            mock_httpx.Client = MagicMock()
            with RastaClient("http://localhost:8020") as client:
                assert client is not None
            client._client.close.assert_called_once()


class TestMaterialIdentification:
    """Test identify_material with known materials."""

    def test_identify_returns_material_result(self):
        """Verify the SDK wraps API response into MaterialResult."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "material": "marble",
            "confidence": 0.95,
            "subcategory": "polished",
            "method": "ollama_vision",
            "model": "llava:latest",
            "properties": {},
        }

        with patch("rasta.sdk.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_httpx.Client.return_value = mock_client

            client = RastaClient()
            # Create a temp file for the test
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(b"\xff\xd8\xff\xe0")  # JPEG header
                tmp = Path(f.name)

            try:
                result = client.identify_material(str(tmp))
                assert isinstance(result, MaterialResult)
                assert result.name == "marble"
                assert result.confidence == 0.95
            finally:
                tmp.unlink()


class TestSDKImports:
    """Verify all public SDK types are importable."""

    def test_all_imports(self):
        from rasta.sdk import (
            AsyncRastaClient,
            FloorPlanResult,
            MaterialResult,
            PipelineResult,
            RastaClient,
            RastaConfig,
            RastaConnectionError,
            RastaError,
            RastaServerError,
            RastaValidationError,
            TextureResult,
        )
        assert all([
            RastaClient, AsyncRastaClient, RastaConfig,
            MaterialResult, TextureResult, PipelineResult, FloorPlanResult,
            RastaError, RastaConnectionError, RastaValidationError, RastaServerError,
        ])
