# Examples

This subdirectory includes a Jupyter notebook example with functionality for `pyINSYDE`. 

To run these examples, it may help to have have [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/) installed. Before you head over to the examples and launch the Jupyter notebooks, you will need to follow these steps:

1) Change your working directory to `examples/` and then run `conda env create -f environment/environment.yml` or replace `conda` with `mamba`. 
2) Activate the environment.
3) Create an ipykernel for the environment. If you are new to Jupyter Notebooks and/or conda, please see: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments. We ran `python -m ipykernel install --user --name insyde_test` or replace the environment name if you updated it. 
4) Change your working directory back to the root of the cloned `pyINSYDE` repository and run `pip install -e .` so that the modules can be imported. 

When you go to the Jupyter notebook to run the example, make sure you activate the `insyde_test` environment. 