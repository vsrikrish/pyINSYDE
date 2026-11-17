
"""Building creation and validated damage calculations for pyINSYDE v2."""

from pyinsyde.insyde import BuildingProperties, Building

from .validation import validate_hazard_inputs


def create_building(building_data: dict) -> Building:
    """Create an INSYDE Building from building properties.

    Args:
        building_data: Dictionary containing building properties.

    Returns:
        A Building object ready for damage calculations.
    """
    properties = BuildingProperties(**building_data)
    return Building(properties)


def compute_damage(building, *, he, v, d, s, q):
    """Validate hazard inputs and calculate building damage.

    Args:
        building: An initialized INSYDE Building.
        he: External flood water height (m).
        v: Water velocity (m/s).
        d: Flood duration (hours).
        s: Sediment concentration.
        q: Binary pollutant indicator (0 or 1).

    Returns:
        Dictionary containing water level, absolute damage,
        relative damage, group damage, and component damage.

    Raises:
        TypeError: If an input has an invalid type.
        ValueError: If an input is outside the accepted range.

    Example:
        >>> building = create_building(building_data)
        >>> result = compute_damage(
        ...     building, he=3, v=0.5, d=24, s=0.05, q=0
        ... )
    """
    validate_hazard_inputs(he=he, v=v, d=d, s=s, q=q)

    return building.compute_damage(
        he=he,
        v=v,
        d=d,
        s=s,
        q=q
    )
