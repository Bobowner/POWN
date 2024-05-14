# Prototypical-Open-World-Node-Classification

## Prerequisits

Python version: 3.10.12
CUDA: 12.4
Dependencies in requirements.txt

## Run the baselines

To reproduce the baseline numbers run the script of the respective baseline:

- run_openwgl_experiment.sh
- run_gcn_experiments.sh
- run_openspectral_experiments.sh
- run_openssl_experiments.sh

## Run POWN

To get the POWN results you need to to run "main_results_pown.py" with a dataset and the number of baysian optimazation steps as count. For example for photo and a thousant optimization steps your run:

- python3 main_results_pown.py --dataset photo --count 1000

## Folders

In experiments you find yml files with a config for each experiment. Not mention parameters are set to the value in default_parameter.


