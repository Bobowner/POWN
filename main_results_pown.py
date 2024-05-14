import wandb
import argparse
import os

import random

from handle_meta_data import load_yml, built_config_with_default
from pathlib import Path
from experiment import run_setting




def run_validation():

   
    wandb.init(project=project_name, mode="online")
    config = built_config_with_default(wandb.config, dataset)
    print(config)
    
    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    
    print("Finished: ", result_dict)



#Define config
path = Path("experiments/pown_final.yml")
sweep_configuration = load_yml(path)

# 3: Start the sweep


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
parser.add_argument('--count', nargs='?', default=100, help="Pass the number of runs for baysian optimization")

ARGS = parser.parse_args()
dataset = ARGS.dataset[0] #"cora"
count = ARGS.count
project_name = 'final_results_pown_'+str(dataset)

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project=project_name)

wandb.agent(sweep_id, function=run_validation, count=count)