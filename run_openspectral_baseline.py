import wandb
import argparse
import os

import random

from handle_meta_data import load_yml, built_config_with_default
from pathlib import Path
from experiment import run_setting


project_name = 'openspectral_baseline_new'

def run_gcn(ARGS):

    dataset = ARGS.dataset[0]
    experiment = "openspectral_baseline_"+str(dataset)+".yml"

    wandb.init(project=project_name, mode="online")
    path = Path("experiments/" + experiment)
    config = load_yml(path)
    print(config)

    config = built_config_with_default(config, dataset)
    
    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)



#run code
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
ARGS = parser.parse_args()
run_gcn(ARGS)