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


def sample_damage(
    building_data: dict,
    distributions: dict,
    he: float,
    v: float,
    d: float,
    s: float,
    q: float,
    n_samples: int = 100,
    rng=None,
) -> np.ndarray:
    """Estimate damage uncertainty caused by missing building inputs."""
    _validate_n_samples(n_samples)

    if rng is None:
        rng = np.random.default_rng()

    damages = []

    for _ in range(n_samples):
        completed = fill_missing_inputs(
            building_data,
            distributions,
            rng=rng,
        )
        building = create_building(completed)
        result = building.compute_damage(he, v, d, s, q)
        damages.append(result["absDamage"])

    return np.asarray(damages, dtype=float)


def summarize_damage(damages: np.ndarray) -> dict:
    """Summarize a nonempty array of finite damage values."""
    try:
        damages = np.asarray(damages, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Damages must be numeric.") from exc

    if damages.ndim != 1:
        raise ValueError("Damages must be a one-dimensional array.")

    if damages.size == 0:
        raise ValueError("Damages cannot be empty.")

    if not np.all(np.isfinite(damages)):
        raise ValueError("Damages must contain only finite values.")

    return {
        "mean": float(np.mean(damages)),
        "std": float(np.std(damages)),
        "min": float(np.min(damages)),
        "max": float(np.max(damages)),
    }
