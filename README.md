# DelayDiscountingPLRNN

This repository contains the data and code for our PLRNN latent factor analysis. We provide here the data and scripts that generate the figures for this project.

## Installation

This software was written for Python 3.12.6 and runs on CPU. We recommend that users install it using the Python venv module for virtual environments:

```
python -m venv path/to/virtual/environment
source path/to/virtual/environment/bin/activate
pip install -r requirements.txt
pip install .
```

## Usage

The data required for this project can be found in the `mean_im_de_trj` directory.

These scripts optimize machine learning models to fit the provided data and visualize the analysis of those models. Some scripts may be run as-is, and some may require command line arguments. Before running these scripts, make sure that the appropriate environment is active. Since some scripts depend on the output of other scripts, those scripts should be run in a certain order, which is addressed below.

### Optimization of the PLRNN

We provide PRNG seeds for the optimization of PLRNN models, and those seeds may be found in the scripts described in this subsection. Main machine learning models can be optimized with the `optimize.py` script. An example command looks like this:

```
python optimize.py --experiment main --model_instance 0 --latent_size 30
```

We produced many model instances for analysis and model selection. This command should be run succesively to accomplish a parameter sweep with argument `--model_instance` taking values 0 through 99 and the argument `--latent_size` taking values 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000. These parameter sweeps are computationally expensive, and we recommend the that the user execute the commands in this document with batch scripts on a high performance computing cluster.

Our shuffling control experiment to evaluate overfitting is performed with the `shuffle_optimize.py` script. An example command looks like this:

```
python shuffle_optimize.py --model_instance 0 --latent_size 30
```

This command should be run succesively to accomplish a parameter sweep with argument `--model_instance` taking values 0 through 99 and the argument `--latent_size` taking values 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000.

The optimization of additional models is required to produce supplemental figures. Our PLRNN regularization selection results are produced with the `regselect_optimize.py` script. An example command looks like this:
```
python regselect_optimize.py --model_instance 0 --LB_factor -5
```

This command should be run succesively to accomplish a parameter sweep with argument `--model_instance` taking values 0 through 99 and the argument `-LB_factor` taking values -8, -7, -6, -5, -4, -3, -2, and -1.

Finally, our LRNN results are produced with the `optimize.py` script. An example command:

```
python optimize.py --experiment linear --model_instance 0 --latent_size 30
```

This command should be run succesively to accomplish a parameter sweep with argument `--model_instance` taking values 0 through 99 and the argument `--latent_size` taking values 3, 5, 10, 20, 30, 40, and 50.


### Producing figures

Our scripts that produce figures should be run in order.

Figures 2 and 3 can be produced by running

```
python performance.py
```

To produce Figures 4 and S3, run the following scripts in order:

```
python observation_model_analysis.py
python observation_model_stats.py
python observation_model_exemplar.py
```

The script `PCA.py` produces Figures 5, 8, 9, and S5:

```
python PCA.py
```

Figure 6 is produced in two parts:

```
python count_states.py
python gyration.py
```

Figure 7 is produced with the following script:
```
python exemplar_PCA.py
```

Figure S2 is produced with the following command:
```
python regselect_performance.py
```

Figure S4 is produced with the following command:
```
python linear_performance.py
```

Figure S1 is produced with the following command:
```
python reward_boxplots.py
```

The main PLRNN figures can be found in `results/main/figures`. The linear RNN supplemental figures can be found in `results/linear/figures`. The regularization selection supplemental figures can be found in `results/regselect/figures`. Figure S1 can be found in `results/figures`.