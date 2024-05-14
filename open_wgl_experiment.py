import random
import torch
import numpy as np
import tensorflow as tf
import tf_geometric as tfg
import wandb

from open_wgl_model import OpenWGL
from open_dataset import load_dataset,load_folds, load_folds_resampling,load_folds_class_variation
from evaluation import eval_validation_model, eval_test_model

from utils import calc_avg_dicts




def run_setting(config, num_repeats = 1, log_all=True):

    if config.fold_type == "resampling":
        datasets = load_folds_resampling(config.dataset, unknown_class_ratio=config.unknown_class_ratio, n_folds = config.n_folds)
    elif config.fold_type == "disjunct":
        datasets = load_folds(config.dataset, unknown_class_ratio=config.unknown_class_ratio)
    elif config.fold_type == "class_var":
        datasets = load_folds_class_variation(config.dataset, unknown_class_ratio=config.unknown_class_ratio)
    else: 
        raise NotImplementedError("The choosen fold type for your exoeriment: " + name + " is not supported")


    if (config.fold_type == "disjunct") | (config.fold_type == "resampling"):    
        result_dict = run_experiment(datasets, config, num_repeats, log_all)
        wandb.log(result_dict)

    elif config.fold_type == "class_var":
        result_dict = run_class_variation_experiment(datasets, config, num_repeats, log_all)

    return result_dict


def run_class_variation_experiment(datasets, config, num_repeats, log_all):
    gpu_number = config.gpu_number
    device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    result_dicts = []
    
    for fold_id, data in enumerate(datasets):
        
        for seed in range(num_repeats):

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
            result_run_dict = perform_one_run(fold_id, seed, data,  config, log_all)
            result_dicts.append(result_run_dict)
            print(result_run_dict)
            wandb.log({"n_known_classes" : data.known_classes.shape[0]})

            wandb.log(result_run_dict)

    return calc_avg_dicts(result_dicts, blacklist=['seed', 'fold_id'])


def run_experiment(datasets, config, num_repeats, log_all):
    
    result_dicts = []

    for seed in range(num_repeats):
        
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        
        for fold_id, data in enumerate(datasets):
    
            result_run_dict = perform_one_run(fold_id, seed, data, config, log_all)
                
            wandb.log(result_run_dict)
            result_dicts.append(result_run_dict)
            
    return calc_avg_dicts(result_dicts, blacklist=['seed', 'fold_id'])

def perform_one_run(fold_id, seed, data,  config, log_all):
    
    owgl_model = OpenWGL(data, config.num_protos , seed=seed, learning_rate = 1e-3, n_epochs=300)

    owgl_model.train_model()

    pred = owgl_model.get_open_predictions()

    device = torch.device('cpu')    
    pred = torch.Tensor(pred)
    
    if (config.fold_type != "class_var"):
        known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi = eval_validation_model(pred,
                                                                                           data.y,
                                                                                           data.known_class_val_mask, 
                                                                                           data.unknown_class_val_mask, 
                                                                                           data.all_class_val_mask, 
                                                                                           device)
    else:
        known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi = (-1.0, -1.0, -1.0, -1.0)
    
    known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi = eval_test_model(pred, 
                                                                                     data.y, 
                                                                                     data.known_class_test_mask, 
                                                                                     data.unknown_class_test_mask, 
                                                                                     data.all_class_test_mask,
                                                                                     device)
    result_run_dict = {
        "fold_id" : fold_id,
        "seed" : seed,
        "known_val_acc_run" : known_val_acc,
        "unknown_val_acc_run" : unknown_val_acc,
        "all_val_acc_run" : all_val_acc,
        "val_unseen_mi_run" : val_unseen_mi,
        "known_test_acc_run": known_test_acc,
        "unknown_test_acc_run": unknown_test_acc,
        "all_test_acc_run" : all_test_acc,
        "test_unseen_mi_run" : test_unseen_mi
    }
    return result_run_dict