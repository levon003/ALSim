# ALSim: Active Learning Simulations

ALSim is a Python library for iterative batch-mode active learning simulations.

## Structure

The package code is stored under `src/alsim`.

To use it, currently `alsim.paths.DERIVED_DATA_DIR` needs to be manually set and `src/alsim/data.py` needs to be updated to load the documents in the expected dictionary format.

## Experiments

This repository was used to generate the results presented in the associated IUI conference paper. The configurations for those experiments can be found in `src/alsim/config.py`.

## History

The ALSim library was solely authored by Zachary Levonian in 2021. It includes code from other MIT-licensed repositories. 

## Citing

Please cite the IUI conference paper if the code in this repository is useful to you in your research. See the CITATION.cff file in this repository.

