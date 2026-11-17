Project Overview
================

What is pyINSYDE?
-----------------

pyINSYDE is a Python implementation of the INSYDE building flood damage
model. It estimates the monetary damage a building may experience under
specified flood hazard conditions.

The model uses two main types of information:

* Building characteristics, such as floor area, building type, and number
  of floors.
* Flood hazard conditions, such as water height, flow velocity, flood
  duration, sediment concentration, and pollutant presence.

Original Model and Version 2
----------------------------

The original implementation contains the core building damage calculations.

The v2 package builds additional functionality around that model, including:

* **Input validation:** Checks that flood hazard inputs satisfy expected
  constraints.
* **Probability distributions:** Provides tools for representing uncertain
  or categorical inputs.
* **Missing input handling:** Identifies and fills missing building data.
* **Uncertainty analysis:** Supports repeated damage calculations under
  sampled inputs.
* **Feature importance:** Helps assess how input variables influence
  estimated damage.

Model Workflow
--------------

A typical workflow using the v2 package is:

#. Prepare building characteristic data.
#. Create a building object.
#. Provide flood hazard conditions.
#. Validate the hazard inputs.
#. Calculate building damage.
#. Analyze uncertainty or input importance when needed.

Damage Outputs
--------------

The model returns several damage outputs:

* **Absolute damage:** Estimated monetary damage.
* **Relative damage:** Damage expressed relative to the building's
  replacement value.
* **Group damage:** Damage organized into broader building categories.
* **Component damage:** Damage estimates for individual building
  components.

The six group damage categories are:

* Cleanup
* Removal
* Non-structural
* Structural
* Finishing
* Systems