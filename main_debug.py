import wandb
import argparse

import torch
import os

import random
import numpy as np

from handle_meta_data import load_yml, built_config_with_default
from experiment import run_setting
from tqdm import tqdm

from pathlib import Path
from collections import namedtuple


def main(ARGS):

    #torch.autograd.set_detect_anomaly(True)
    path = Path("experiments/debug.yml")
    config = load_yml(path)
    run = wandb.init(project='debug', mode="disabled", config=config)
    #config = namedtuple('Config', config.keys())(**config)
        
    config = built_config_with_default(wandb.config, config["dataset"])
    print(config)

    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)


#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
#ARGS = parser.parse_args()
ARGS = ""
main(ARGS)