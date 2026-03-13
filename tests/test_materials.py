"""Tests for material identification and properties."""

import pytest

from rasta.texture_identify import MATERIALS, list_materials


class TestMaterialsDatabase:
    """Validate the materials database integrity."""

    def test_23_materials(self):
        assert len(MATERIALS) == 23

    def test_all_materials_have_required_fields(self):
        required = {"subcategory", "thickness_mm", "rf_attenuation_db", "thermal_conductivity", "osm_tags"}
        for name, props in MATERIALS.items():
            missing = required - set(props.keys())
            assert not missing, f"{name} missing fields: {missing}"

    def test_rf_attenuation_has_three_bands(self):
        bands = {"sub_1ghz", "2_4ghz", "5ghz"}
        for name, props in MATERIALS.items():
            rf = props["rf_attenuation_db"]
            missing = bands - set(rf.keys())
            assert not missing, f"{name} RF missing bands: {missing}"

    def test_attenuation_increases_with_frequency(self):
        """Higher frequency = more attenuation for all materials."""
        for name, props in MATERIALS.items():
            rf = props["rf_attenuation_db"]
            assert rf["sub_1ghz"] <= rf["2_4ghz"] <= rf["5ghz"], (
                f"{name}: attenuation should increase with frequency"
            )

    def test_thickness_positive(self):
        for name, props in MATERIALS.items():
            assert props["thickness_mm"] > 0, f"{name} has non-positive thickness"

    def test_thermal_conductivity_positive(self):
        for name, props in MATERIALS.items():
            assert props["thermal_conductivity"] > 0, f"{name} has non-positive thermal conductivity"

    def test_metals_highest_attenuation(self):
        """Metal should have the highest RF attenuation."""
        metal_atten = MATERIALS["metal_sheet"]["rf_attenuation_db"]["2_4ghz"]
        for name, props in MATERIALS.items():
            if not name.startswith("metal"):
                assert props["rf_attenuation_db"]["2_4ghz"] <= metal_atten, (
                    f"{name} has higher 2.4GHz attenuation than metal_sheet"
                )

    def test_list_materials_returns_all(self):
        materials = list_materials()
        assert len(materials) == 23
        names = {m["name"] for m in materials}
        assert names == set(MATERIALS.keys())
