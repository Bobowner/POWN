import wandb
import argparse
import os

import random

from handle_meta_data import load_yml, built_config_with_default
from pathlib import Path
from experiment import run_setting


project_name = 'debug'
dataset = "cora"

def run_validation():

   
    wandb.init(project=project_name, mode="online")
    config = built_config_with_default(wandb.config, dataset)
    print(config)
    
    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    
    print("Finished: ", result_dict)



#Define config
path = Path("experiments/check_loss_weights_cora_baysian.yml")
sweep_configuration = load_yml(path)

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project=project_name)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
ARGS = parser.parse_args()

wandb.agent(sweep_id, function=run_validation, count=50)