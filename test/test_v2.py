import unittest
import numpy as np

from pyinsyde.v2.missing_inputs import (
    find_missing_inputs,
    fill_missing_inputs,
)
from pyinsyde.v2.distributions import (
    CategoricalDistribution,
    ContinuousDistribution,
    EmpiricalDistribution,
)
from pyinsyde.v2.uncertainty import (
    sample_damage,
    summarize_damage,
)
from pyinsyde.v2.feature_importance import (
    analyze_missing_input,
    compare_missing_inputs,
    rank_missing_inputs,
)


class TestDistributions(unittest.TestCase):
    def test_categorical_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution(["a", "b"], [0.7, 0.7])

        with self.assertRaises(ValueError):
            CategoricalDistribution(["a", "b"], [-0.1, 1.1])

    def test_continuous_rejects_invalid_standard_deviation(self):
        with self.assertRaises(ValueError):
            ContinuousDistribution(mean=5, sd=0)

        with self.assertRaises(ValueError):
            ContinuousDistribution(mean=5, sd=-1)

    def test_continuous_rejects_reversed_bounds(self):
        with self.assertRaises(ValueError):
            ContinuousDistribution(
                mean=5,
                sd=1,
                lower=10,
                upper=2,
            )

    def test_empirical_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            EmpiricalDistribution([])

    def test_continuous_distribution_respects_bounds_far_from_mean(self):
        distribution = ContinuousDistribution(
            mean=0,
            sd=1,
            lower=10,
            upper=11,
        )
        rng = np.random.default_rng(42)

        for _ in range(100):
            value = distribution.sample(rng)
            self.assertGreaterEqual(value, 10)
            self.assertLessEqual(value, 11)

    def test_continuous_distribution_with_equal_bounds(self):
        distribution = ContinuousDistribution(
            mean=0,
            sd=1,
            lower=5,
            upper=5,
        )
        self.assertEqual(distribution.sample(), 5)

    def test_continuous_distribution_rejects_nonnumeric_mean(self):
        with self.assertRaises(ValueError):
            ContinuousDistribution(mean="not a number", sd=1)

    def test_categorical_distribution_rejects_nonnumeric_probabilities(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution(
                values=["low", "high"],
                probabilities=["invalid", 1],
            )


class TestMissingInputs(unittest.TestCase):
    def test_find_missing_inputs(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": None,
            "NF": 2,
            "BT": 2,
            "BS": None,
            "PD": 1,
            "PT": 1,
            "YY": None,
            "random_variable": None,
        }

        missing = find_missing_inputs(building_data)

        self.assertEqual(set(missing), {"BH", "BS", "YY"})

    def test_find_missing_required_fields_that_are_omitted(self):
        building_data = {
            "FA": 150,
            "EP": 50,
        }

        missing = find_missing_inputs(building_data)

        self.assertEqual(
            set(missing),
            {"IH", "BH", "NF", "BT", "BS", "PD", "PT", "YY"},
        )

    def test_omitted_optional_defaults_are_not_missing(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        self.assertEqual(find_missing_inputs(building_data), [])

    def test_optional_derived_values_are_not_sampled(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
            "IA": None,
            "BE": None,
            "BA": None,
        }

        completed = fill_missing_inputs(
            building_data,
            {},
            rng=np.random.default_rng(42),
        )

        self.assertIsNone(completed["IA"])
        self.assertIsNone(completed["BE"])
        self.assertIsNone(completed["BA"])
        self.assertEqual(completed["FA"], 150)

    def test_fill_missing_inputs(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": None,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        distributions = {
            "BH": EmpiricalDistribution([2.0, 2.5, 3.0]),
        }

        completed = fill_missing_inputs(
            building_data,
            distributions,
            rng=np.random.default_rng(42),
        )

        self.assertIn(completed["BH"], [2.0, 2.5, 3.0])
        self.assertEqual(completed["FA"], 150)
        self.assertIsNone(building_data["BH"])

    def test_missing_distribution_raises_error(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": None,
        }

        with self.assertRaises(ValueError):
            fill_missing_inputs(
                building_data,
                {},
                rng=np.random.default_rng(42),
            )


class TestFeatureImportance(unittest.TestCase):
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
            "YY": 1980,
        }

        self.distribution = EmpiricalDistribution(
            [100, 150, 200, 250, 300]
        )

        self.hazard = {
            "he": 3,
            "v": 0.5,
            "d": 24,
            "s": 0.05,
            "q": 1,
        }

    def test_rank_missing_inputs(self):
        results = [
            {"input": "IH", "variance": 100.0},
            {"input": "FA", "variance": 500.0},
            {"input": "BT", "variance": 250.0},
        ]

        ranked = rank_missing_inputs(results)

        self.assertEqual(
            [result["input"] for result in ranked],
            ["FA", "BT", "IH"],
        )
        self.assertEqual(results[0]["input"], "IH")

    def test_analyze_missing_input(self):
        result = analyze_missing_input(
            self.building_data,
            "FA",
            self.distribution,
            **self.hazard,
            n_samples=10,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(result["input"], "FA")
        self.assertEqual(result["baseline_damage"], 41606.6)
        self.assertGreater(result["std_damage"], 0)
        self.assertLessEqual(result["min_damage"], result["max_damage"])

    def test_analyze_missing_input_rejects_zero_samples(self):
        with self.assertRaises(ValueError):
            analyze_missing_input(
                self.building_data,
                "FA",
                self.distribution,
                **self.hazard,
                n_samples=0,
            )

    def test_analyze_missing_input_rejects_invalid_sample_counts(self):
        for invalid_count in (0, -1, 2.5, True):
            with self.subTest(n_samples=invalid_count):
                with self.assertRaises(ValueError):
                    analyze_missing_input(
                        {},
                        "FA",
                        EmpiricalDistribution([100, 150, 200]),
                        **self.hazard,
                        n_samples=invalid_count,
                    )

    def test_compare_missing_inputs_rejects_invalid_sample_counts(self):
        distributions = {"FA": self.distribution}

        for invalid_count in (0, -1, 2.5, True):
            with self.subTest(n_samples=invalid_count):
                with self.assertRaises(ValueError):
                    compare_missing_inputs(
                        {},
                        distributions,
                        **self.hazard,
                        n_samples=invalid_count,
                    )


class TestUncertainty(unittest.TestCase):
    def test_summarize_damage(self):
        damages = np.array([100.0, 200.0, 300.0])
        summary = summarize_damage(damages)

        self.assertAlmostEqual(summary["mean"], 200.0)
        self.assertAlmostEqual(summary["std"], np.std(damages))
        self.assertAlmostEqual(summary["min"], 100.0)
        self.assertAlmostEqual(summary["max"], 300.0)

    def test_sample_damage_count(self):
        building_data = {
            "FA": None,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        distributions = {
            "FA": EmpiricalDistribution([100, 150, 200, 250, 300]),
        }

        damages = sample_damage(
            building_data,
            distributions,
            he=3,
            v=0.5,
            d=24,
            s=0.05,
            q=1,
            n_samples=10,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(len(damages), 10)
        self.assertTrue(np.all(np.isfinite(damages)))

    def test_sample_damage_is_reproducible(self):
        building_data = {
            "FA": None,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        distributions = {
            "FA": EmpiricalDistribution([100, 150, 200, 250, 300]),
        }

        damages_1 = sample_damage(
            building_data,
            distributions,
            he=3,
            v=0.5,
            d=24,
            s=0.05,
            q=1,
            n_samples=10,
            rng=np.random.default_rng(42),
        )

        damages_2 = sample_damage(
            building_data,
            distributions,
            he=3,
            v=0.5,
            d=24,
            s=0.05,
            q=1,
            n_samples=10,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(damages_1, damages_2)

    def test_sample_damage_rejects_zero_samples(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        with self.assertRaises(ValueError):
            sample_damage(
                building_data,
                {},
                he=3,
                v=0.5,
                d=24,
                s=0.05,
                q=1,
                n_samples=0,
            )

    def test_sample_damage_rejects_invalid_sample_counts(self):
        building_data = {
            "FA": 150,
            "EP": 50,
            "IH": 3.0,
            "BH": 2.0,
            "NF": 2,
            "BT": 2,
            "BS": 1,
            "PD": 1,
            "PT": 1,
            "YY": 1980,
        }

        distributions = {
            "FA": EmpiricalDistribution([100, 150, 200]),
            "IH": EmpiricalDistribution([2.5, 3.0, 3.5]),
        }

        for invalid_count in (0, -1, 2.5, True):
            with self.subTest(n_samples=invalid_count):
                with self.assertRaises(ValueError):
                    sample_damage(
                        building_data,
                        distributions,
                        he=3,
                        v=0.5,
                        d=24,
                        s=0.05,
                        q=1,
                        n_samples=invalid_count,
                    )

    def test_summarize_damage_rejects_empty_array(self):
        with self.assertRaises(ValueError):
            summarize_damage(np.array([]))

    def test_summarize_damage_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            summarize_damage(np.array([100.0, np.nan]))

    def test_summarize_damage_rejects_multidimensional_array(self):
        with self.assertRaises(ValueError):
            summarize_damage(np.array([[100.0, 200.0]]))


if __name__ == "__main__":
    unittest.main()
