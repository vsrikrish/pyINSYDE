import numpy as np

from pyinsyde.v2.missing_inputs import fill_missing_inputs
from pyinsyde.v2.model import create_building


def _validate_n_samples(n_samples):
    """Require n_samples to be a positive integer, excluding booleans."""
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples <= 0
    ):
        raise ValueError("n_samples must be a positive integer.")


def analyze_missing_input(
    building_data: dict,
    input_name: str,
    distribution,
    he: float,
    v: float,
    d: float,
    s: float,
    q: float,
    n_samples: int = 100,
    rng=None,
) -> dict:
    """Measure how uncertainty in one missing input affects damage."""
    _validate_n_samples(n_samples)

    if rng is None:
        rng = np.random.default_rng()

    baseline_building = create_building(building_data)

    baseline_result = baseline_building.compute_damage(
        he,
        v,
        d,
        s,
        q,
    )

    baseline_damage = baseline_result["absDamage"]

    missing_data = building_data.copy()
    missing_data[input_name] = None

    damages = []

    for _ in range(n_samples):
        completed = fill_missing_inputs(
            missing_data,
            {input_name: distribution},
            rng=rng,
        )

        building = create_building(completed)

        result = building.compute_damage(
            he,
            v,
            d,
            s,
            q,
        )

        damages.append(result["absDamage"])

    damages = np.asarray(damages, dtype=float)
    differences = damages - baseline_damage

    return {
        "input": input_name,
        "baseline_damage": baseline_damage,
        "mean_damage": float(np.mean(damages)),
        "std_damage": float(np.std(damages)),
        "variance": float(np.var(damages)),
        "mean_difference": float(np.mean(differences)),
        "mean_absolute_difference": float(
            np.mean(np.abs(differences))
        ),
        "min_damage": float(np.min(damages)),
        "max_damage": float(np.max(damages)),
    }


def compare_missing_inputs(
    building_data: dict,
    distributions: dict,
    he: float,
    v: float,
    d: float,
    s: float,
    q: float,
    n_samples: int = 100,
    rng=None,
) -> list:
    """Compare the effect of uncertainty across multiple missing inputs."""
    _validate_n_samples(n_samples)

    if rng is None:
        rng = np.random.default_rng()

    results = []

    for input_name, distribution in distributions.items():
        result = analyze_missing_input(
            building_data,
            input_name,
            distribution,
            he,
            v,
            d,
            s,
            q,
            n_samples=n_samples,
            rng=rng,
        )

        results.append(result)

    return results


def rank_missing_inputs(results: list) -> list:
    """Rank missing building inputs by their contribution to damage variance.

    Args:
        results: List of result dictionaries returned by
            compare_missing_inputs().

    Returns:
        A new list sorted from highest to lowest damage variance.
    """
    return sorted(
        results,
        key=lambda result: result["variance"],
        reverse=True,
    )
