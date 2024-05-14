import argparse
import wandb
import torch

from handle_meta_data import load_yml, built_config_with_default
from models.n_class_estimator import estimate_number_of_classes
from open_dataset import load_dataset,load_folds, load_folds_resampling, load_folds_class_variation
from pathlib import Path


def main(ARGS):

    project_name ="class_estimation"
    #torch.autograd.set_detect_anomaly(True)
    dataset = ARGS.dataset[0]
    path = Path("experiments/n_class_estimation" + "_" + dataset +".yml")
    config = load_yml(path)
    run = wandb.init(project=project_name, mode="online", config=config)
    #config = namedtuple('Config', config.keys())(**config)

    #config["expname"] = p.name
    config = built_config_with_default(wandb.config, config["dataset"])
    print(config)

    result_dict = run_setting(config, num_repeats = config.num_repeats, log_all=config.log_all)
    print("Finished: ", result_dict)


def run_setting(config, num_repeats = 1, log_all=False):
    datasets = load_folds(config.dataset, unknown_class_ratio=config.unknown_class_ratio)
    device = torch.device(*('cuda', config.gpu_number) if torch.cuda.is_available() else 'cpu')

    for data in datasets:
        n_classes = estimate_number_of_classes(data, device, config)
        wandb.log({'Best k': n_classes})
        print(n_classes)

    


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
ARGS = parser.parse_args()
main(ARGS)      