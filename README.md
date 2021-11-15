# Transient Stability Analysis with Physics-Informed Neural Networks

This repository is the official implementation of [Transient Stability Analysis with Physics-Informed Neural Networks](https://arxiv.org/abs/2106.13638). 

## Environment

To install and activate the environment using conda run:

```setup
conda env create -f environment.yml
conda activate pinns_tf_2_4
```

##  Code structure
The code is structured in the following way:
- `train_model.py` contains the entire workflow to train a single model
- `power_system_functions.py` sets up the power system model, including the parameters and the relevant state equations for simulations and the physics evaluations within the PINN.
- `PINN.py` defines the neural network model that inherits from the class `tensorflow.keras.models.Model`
- `create_data.py` creates a database of trajectories that is used in the selection of the training, validation, and test data. Needs to be run only once.
- `dataset_handling.py` prepares the data by splitting them and provide the correct format.
- `setup_and_run` provides a wrapper to setup and run multiple training processes in parallel.

## Folder structure
The directory for the storage of all data should contain the following folders and needs to be defined in `train_model.py`, `create_data.py`, and `setup_and_run`:
- `datasets`
- `logs`
- `result_datasets`
- `model_weights`
- `quantiles`
- `setup_tables`

## Citation

@misc{stiasny2021transient,
      title={Transient Stability Analysis with Physics-Informed Neural Networks}, 
      author={Jochen Stiasny and Georgios S. Misyris and Spyros Chatzivasileiadis},
      year={2021},
      eprint={2106.13638},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
 
 ## Related work
 
The concept of PINNs was introduced by Raissi et al. (https://maziarraissi.github.io/PINNs/) and adapted to power systems by Misyris et al. (https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems). The presented code is inspired by these two sources.
