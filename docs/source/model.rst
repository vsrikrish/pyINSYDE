Model Usage
===========

The v2 model module provides functions for creating building objects and
calculating flood damage.

Creating a Building
-------------------

Use ``create_building()`` to create a building from a dictionary of
building characteristics.

.. code-block:: python

   from pyinsyde.v2 import create_building

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
       "YY": 1980
   }

   building = create_building(building_data)

The dictionary keys must correspond to the fields accepted by
``BuildingProperties`` in the original implementation.

Calculating Damage
------------------

Use ``compute_damage()`` to calculate damage for a building under specified
flood conditions.

.. code-block:: python

   from pyinsyde.v2 import compute_damage

   results = compute_damage(
       building,
       he=3.0,
       v=0.5,
       d=24,
       s=0.05,
       q=0
   )

The function returns a dictionary containing:

* ``waterLevel``
* ``absDamage``
* ``relDamage``
* ``groupDamage``
* ``componentDamage``

Understanding the Inputs
------------------------

The hazard inputs represent:

* ``he``: External water height.
* ``v``: Flow velocity.
* ``d``: Flood duration.
* ``s``: Sediment concentration.
* ``q``: Binary indicator for pollutant presence.

For the accepted input constraints, see :doc:`validation`.