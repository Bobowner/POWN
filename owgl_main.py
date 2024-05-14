import wandb
import argparse

import torch

import random
import numpy as np

from handle_meta_data import load_yml, built_config_with_default
from open_wgl_experiment import run_setting
from tqdm import tqdm

from pathlib import Path
from collections import namedtuple


#experiment = "openwgl_baseline_"+str(dataset)+".yml"
project_name = 'openwgl_baseline'

def main(ARGS):
    
    dataset = ARGS.dataset[0]
    experiment = "openwgl_baseline_resampling_"+str(dataset)+".yml"
    path = Path("experiments/" + experiment)

    config = load_yml(path)
    run = wandb.init(project=project_name, mode="online", config=config)
    #config = namedtuple('Config', config.keys())(**config)
    
    config = built_config_with_default(wandb.config, ARGS.dataset[0])
    print(config)

    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
ARGS = parser.parse_args()
main(ARGS)