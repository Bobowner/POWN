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


def main():

    #torch.autograd.set_detect_anomaly(True)
    run = wandb.init(project=project_name, mode="online")
    #config = namedtuple('Config', config.keys())(**config)
        
    config = built_config_with_default(wandb.config, dataset)
    print(config)

    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
parser.add_argument('--parameter',nargs='+', default="suplossweight", help="Pass name of the dataset you use")

ARGS = parser.parse_args()
dataset = ARGS.dataset[0]
parameter =  ARGS.parameter[0]
project_name ='hyper_parameter_sensitivity_'+parameter+"_"+dataset

path = Path("experiments/pown_hp_sensitivity_"+parameter+"_" +dataset+".yml")
sweep_configuration = load_yml(path)


ARGS = ""
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project=project_name)

wandb.agent(sweep_id, function=main)