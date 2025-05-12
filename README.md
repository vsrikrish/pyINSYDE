[![Version](https://img.shields.io/badge/version-0.1-blue.svg)](https://github.com/vsrikrish/pyinsyde)
[![DOI](https://zenodo.org/badge/980586904.svg)](https://doi.org/10.5281/zenodo.15388700)
[![License](https://img.shields.io/badge/License-BSD--2--Clause-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintenance-Active-brightgreen.svg)](Maintenance)

# pyINSYDE

**pyINSYDE**

Python port of the INSYDE synthetic flood damage model (Dottori et al, 2016).

\* original author: Cella Schnabel, rms428@cornell.edu
\* corresponding author:  Vivek Srikrishnan, vs498@cornell.edu

## Purpose

Estimating damages from flooding are often based on simple, deterministic depth-damage functions (DDFs) which rely on historical claims or reporting data and expert assessments. While large-scale damage estimates using DDFs are readily scalable, the underlying estimation methods can be subject to input data errors or the typical potential biases associated with expert assessments. While aggregate damage estimates may be robust to these errors, they may matter more for individual properties. Moreover, there may be considerable uncertainties associated with how both building characteristics and flood dynamics influence the depth-damage relationship.

INSYDE is a synthetic damage model using a reduced-form but mechanistically-motivated representation of building damages and associated repair and recovery costs. The goal of `pyINSYDE` is to provide a Python port of this model to faciliate assessments of depth-damage uncertainty.

## Journal reference

Original INSYDE reference:

Dottori, F., Figueiredo, R., Martina, M. L. V., Molinari, D., & Scorzini, A. R. (2016). INSYDE: a synthetic, probabilistic flood damage model based on explicit cost analysis. *Natural Hazards and Earth System Sciences*, 16(12), 2577–2591. <https://doi.org/10.5194/nhess-16-2577-2016>

## Installation

### Dependencies

`pyINSYDE` was developed using Python 3.9.

| Package | Version |
|-------|---------|
| `numpy` | `>=2.0` |
| `scipy` | `>=1.9` |

### Instructions

#### For Users

To use pyINSYDE, install using `pip install git+https://github.com/vsrikrish/pyinsyde`. We recommend installing `pyINSYDE` in a virtual environment.


#### For Developers

If you would like to edit the source code:

1. Clone the repository:

   ```bash
   git clone https://github.com/vsrikrish/pyinsyde.git
   cd pyinsyde
   ```
2. Install the dependencies above (ideally in a virtual environment). `pyINSYDE` is designed to be lightweight and should work with most versions of Python 3, NumPy, and SciPy. If you use older versions, you can test functionality with `test/insyde_example.py`.
3. Install `pyINSYDE` in development mode: `pip install -e`.

## Getting Started

### Buildings

pyINSYDE is structured around `Building` objects, which can then be subjected to a variety flood events. To create a new `Building`, construct a dictionary containing the relevant attributes and unpack this dictionary into a `BuildingProperties` dataclass. Finally, call the `Building` constructor using the `BuildingProperties` object.

Example:

```python
from pyinsyde import BuildingProperties, Building

property_dict = ...
bp = BuildingProperties(**property_dict)
b = Building(bp)
```

A `BuildingProperties` object can also be constructed without the initial dictionary, but this approach increases the flexibility of specifying attributes.

In addition to the structural characteristics for the building, there are a few optional settings which can be set in the initial dictionary. 

* Specifying a value for `sd` will add uncertainty to the component-wide damage estimates based on that number of standard deviations relative to the associated fragility curve range.
* By default, pyINSYDE damage accounting will use the replacement value and unit price data from the original INSYDE model. These can be replaced by passing (absolute) paths as `rp_val` and `up_val`. We assume that these will have the same formatting as the original INSYDE data (found in `data/replacement_values.txt` and `data/unit_prices.txt`).

## Usage Examples

Coming...

## Documentation

Coming...

## Contribution Guidelines

Contributions are welcome, especially those aimed at increasing the flexibility of pyINSYDE to new locations or datasets! Please fork, submit pull requests, or report bugs.

## License

pyINSYDE uses the [GNU GPLv3 license](LICENSE). For more details, see [GNU GPLv3 on Choose A License](https://choosealicense.com/licenses/gpl-3.0/#).