
import unittest

from pyinsyde.v2.validation import validate_hazard_inputs
from pyinsyde.v2.model import compute_damage


class TestHazardValidation(unittest.TestCase):

    def test_valid_inputs(self):
        validate_hazard_inputs(
            he=3, v=0.5, d=24, s=0.05, q=0
        )

    def test_boolean_pollution_inputs(self):
        validate_hazard_inputs(
            he=3, v=0.5, d=24, s=0.05, q=False
        )
        validate_hazard_inputs(
            he=3, v=0.5, d=24, s=0.05, q=True
        )

    def test_invalid_pollution_values(self):
        for q in [-1, 0.5, 2]:
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    validate_hazard_inputs(
                        he=3, v=0.5, d=24, s=0.05, q=q
                    )

    def test_negative_hazard_inputs(self):
        inputs = {
            "he": -1,
            "v": -1,
            "d": -1,
            "s": -1,
        }

        for name, value in inputs.items():
            with self.subTest(name=name):
                arguments = {
                    "he": 3,
                    "v": 0.5,
                    "d": 24,
                    "s": 0.05,
                    "q": 0,
                }
                arguments[name] = value

                with self.assertRaises(ValueError):
                    validate_hazard_inputs(**arguments)

    def test_nonfinite_inputs(self):
        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_hazard_inputs(
                        he=value, v=0.5, d=24, s=0.05, q=0
                    )

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            validate_hazard_inputs(
                he="3", v=0.5, d=24, s=0.05, q=0
            )

    def test_damage_wrapper_validates_before_calculation(self):
        class DummyBuilding:
            def compute_damage(self, **kwargs):
                return {"called": True}

        building = DummyBuilding()

        result = compute_damage(
            building, he=3, v=0.5, d=24, s=0.05, q=0
        )

        self.assertEqual(result, {"called": True})

    def test_damage_wrapper_rejects_invalid_inputs(self):
        class DummyBuilding:
            def compute_damage(self, **kwargs):
                raise AssertionError("Damage calculation should not run.")

        building = DummyBuilding()

        with self.assertRaises(ValueError):
            compute_damage(
                building, he=3, v=0.5, d=24, s=0.05, q=2
            )


if __name__ == "__main__":
    unittest.main()
