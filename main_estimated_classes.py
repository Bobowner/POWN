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
    path = Path("experiments/estimated_n_class_" + model + "_" + dataset +".yml")
    config = load_yml(path)
    run = wandb.init(project=project_name, mode="online", config=config)
    #config = namedtuple('Config', config.keys())(**config)

    #config["expname"] = p.name
    config = built_config_with_default(wandb.config, config["dataset"])
    print(config)

    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
parser.add_argument('--model',nargs='+', default=None, help="Pass name of the model you use")
ARGS = parser.parse_args()
dataset = ARGS.dataset[0]
model =  ARGS.model[0]
project_name = "class_estimation"

main(ARGS)