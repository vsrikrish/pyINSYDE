
"""Input validation for the pyINSYDE v2 damage model."""

import math
from numbers import Real


def validate_hazard_inputs(*, he, v, d, s, q):
    """Validate inputs supplied to the flood damage model.

    Args:
        he: External flood water height (m).
        v: Water velocity (m/s).
        d: Flood duration (hours).
        s: Sediment concentration.
        q: Binary pollutant indicator (0 or 1).

    Raises:
        TypeError: If a numerical input is not a real number.
        ValueError: If an input is non-finite, negative, or q is not binary.

    Notes:
        The original INSYDE model documents q as a binary indicator.
        This function accepts booleans and numerical 0/1 values.
    """

    continuous_inputs = {
        "he": he,
        "v": v,
        "d": d,
        "s": s,
    }

    for name, value in continuous_inputs.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")

    if isinstance(q, bool):
        return

    if not isinstance(q, Real):
        raise TypeError("q must be a boolean or numeric 0 or 1.")

    if not math.isfinite(float(q)):
        raise ValueError("q must be finite.")

    if q not in (0, 1):
        raise ValueError("q must be binary: 0 or 1.")
