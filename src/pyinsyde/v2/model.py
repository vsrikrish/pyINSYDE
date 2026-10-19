from pyinsyde.insyde import BuildingProperties, Building


def create_building(building_data: dict) -> Building:
    """Create an INSYDE Building from completed building data.

    Args:
        building_data: Dictionary containing complete building properties.

    Returns:
        A Building object ready for damage calculations.
    """
    properties = BuildingProperties(**building_data)
    return Building(properties)