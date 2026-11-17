import unittest
import numpy as np

from pyinsyde.v2.model import create_building


class TestModel(unittest.TestCase):

    def setUp(self):
        self.building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980
        }

        self.building = create_building(self.building_data)

    def test_create_building(self):
        self.assertEqual(self.building.FA, 150)
        self.assertEqual(self.building.NF, 2)

    def test_optional_defaults(self):
        self.assertFalse(self.building.BE)
        self.assertEqual(self.building.IA, 135)
        self.assertEqual(self.building.BA, 0)

    def test_water_level(self):
        self.assertEqual(self.building.waterLevel(3), 3)
        self.assertEqual(self.building.waterLevel(-1), 0)
        self.assertEqual(self.building.waterLevel(100), 6.3)

    def test_damage_output_structure(self):
        result = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        expected_keys = {
            "waterLevel",
            "absDamage",
            "relDamage",
            "groupDamage",
            "componentDamage"
        }

        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(len(result["groupDamage"]), 6)
        self.assertEqual(len(result["componentDamage"]), 32)

    def test_baseline_damage(self):
        result = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        self.assertAlmostEqual(result["waterLevel"], 3.0)
        self.assertAlmostEqual(result["absDamage"], 41606.6)
        self.assertAlmostEqual(
            result["relDamage"], 0.09684962756052141
        )

    def test_damage_values_are_finite_and_nonnegative(self):
        result = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        values = [
            result["absDamage"],
            result["relDamage"],
            *result["groupDamage"].values(),
            *result["componentDamage"].values()
        ]

        for value in values:
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0)

    def test_group_damage_sums_to_total(self):
        result = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        group_total = sum(result["groupDamage"].values())

        self.assertAlmostEqual(
            group_total, result["absDamage"]
        )

    def test_repeated_calculations_are_consistent(self):
        result1 = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        result2 = self.building.compute_damage(
            he=3, v=0.5, d=24, s=0.05, q=1
        )

        self.assertAlmostEqual(
            result1["absDamage"], result2["absDamage"]
        )


if __name__ == "__main__":
    unittest.main()
