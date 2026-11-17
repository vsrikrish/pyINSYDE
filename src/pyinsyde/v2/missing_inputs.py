
from dataclasses import MISSING, fields

from pyinsyde.insyde import BuildingProperties


def find_missing_inputs(building_data: dict) -> list:
    """Identify missing required building characteristics.

    Required properties are missing if they are omitted from the input
    dictionary or explicitly set to None.

    Optional properties retain the original INSYDE default and
    derivation behavior.

    Args:
        building_data: Dictionary containing building properties.

    Returns:
        List of missing required property names.
    """
    required_fields = [
        field.name
        for field in fields(BuildingProperties)
        if field.default is MISSING
        and field.default_factory is MISSING
    ]

    return [
        name
        for name in required_fields
        if name not in building_data or building_data[name] is None
    ]


def fill_missing_inputs(
    building_data: dict,
    distributions: dict,
    rng=None,
) -> dict:
    """Fill missing required building inputs using their distributions.

    Optional properties are left to the original INSYDE model to
    default or derive as appropriate.

    Args:
        building_data: Dictionary containing building properties.
        distributions: Distributions for missing required properties.
        rng: Optional random number generator.

    Returns:
        A new dictionary with missing required inputs filled in.
    """
    completed_data = building_data.copy()

    missing_inputs = find_missing_inputs(building_data)

    for name in missing_inputs:
        if name not in distributions:
            raise ValueError(
                f"No distribution provided for missing input: {name}"
            )

        completed_data[name] = distributions[name].sample(rng)

    # Explicit None values for these optional properties would override
    # their dataclass defaults, so omit them and let INSYDE apply defaults.
    for name in ("GL", "FL", "LM", "sd"):
        if completed_data.get(name) is None:
            completed_data.pop(name, None)

    return completed_data
