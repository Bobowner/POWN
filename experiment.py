import yaml
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import random


from results_writer import write_scores, write_threshold_scores

from torch_geometric.nn.models import GCN
from torch_geometric.loader import NeighborLoader

from open_dataset import load_dataset,load_folds, load_folds_resampling, load_folds_class_variation
from model_setup import setup_model
from evaluation import eval_validation_model, eval_test_model
from utils import mean_and_std_tensor_list
from train_model import train, train_with_sampler, eval_on_cpu, eval_with_sampler

from utils import calc_avg_dicts


def run_setting(config, num_repeats = 1, log_all=True):


    if config.fold_type == "resampling":
        datasets = load_folds_resampling(config.dataset, unknown_class_ratio=config.unknown_class_ratio, n_folds = config.n_folds)
    elif (config.fold_type == "disjunct") | (config.fold_type == "n_class"):
        
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
    
    elif config.fold_type == "n_class":
        result_dict = run_estimated_class_numbers_experiment(datasets, config, num_repeats, log_all)
    else:
        raise NotImplementedError("The choosen setting for your exoeriment: " + name + " is not supported")


    return result_dict


def run_estimated_class_numbers_experiment(datasets, config, num_repeats, log_all):
    
    gpu_number = config.gpu_number
    device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
    print(device)
    #device = torch.device('cpu')

    result_dicts = []

    for seed in range(num_repeats):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        for fold_id, (data, num_protos) in enumerate(zip(datasets, config.num_protos)):

            print(num_protos)
            result_run_dict = perform_one_run(fold_id, seed, data, device, config, log_all, num_protos=num_protos)
                
            wandb.log(result_run_dict)
            result_dicts.append(result_run_dict)
            
    return calc_avg_dicts(result_dicts, blacklist=['seed', 'fold_id'])
    

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

    
            
            result_run_dict = perform_one_run(fold_id, seed, data, device, config, log_all, num_protos=config.num_protos) 
            result_dicts.append(result_run_dict)
            print(result_run_dict)
            wandb.log({"n_known_classes" : data.known_classes.shape[0]})

            wandb.log(result_run_dict)

    return calc_avg_dicts(result_dicts, blacklist=['seed', 'fold_id'])



    
    

def run_experiment(datasets, config, num_repeats, log_all):

    gpu_number = config.gpu_number
    device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    result_dicts = []

    for seed in range(num_repeats):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        for fold_id, data in enumerate(datasets):
    
            result_run_dict = perform_one_run(fold_id, seed, data, device, config, log_all, num_protos=config.num_protos)
                
            wandb.log(result_run_dict)
            result_dicts.append(result_run_dict)
            
    return calc_avg_dicts(result_dicts, blacklist=['seed', 'fold_id'])



def perform_one_run(fold_id, seed, data, device, config, log_all, num_protos):
    
    og_model = setup_model(data, device, config, num_protos, log_all=log_all)

    if config.model != "spectral":
        optimizer = torch.optim.Adam(og_model.parameters(), 
                                             lr=config.lr, 
                                             weight_decay=config.weight_decay)
    else:
        optimizer = None
    
    if config.sampler == "neighbor":
        og_model = train_with_sampler(og_model, optimizer, data, device, log_all, config)
    elif config.sampler == None:
        og_model = train(og_model, optimizer, data, device, log_all, config)
    else:
        raise NotImplementedError("Your sampler is not supported yet!")

    og_model.eval()
    data = data.to(device)

    with torch.no_grad():
        if config.sampler == "neighbor":
            eval_result = eval_on_cpu(og_model, data, device, config)
            #eval_result = eval_with_sampler(og_model=og_model, data=data, device=device, config=config)
            known_val_acc = eval_result[0]
            unknown_val_acc = eval_result[1]
            all_val_acc = eval_result[2]
            val_unseen_mi = eval_result[3]
            known_val_mat = eval_result[4]
            known_test_acc = eval_result[6]
            unknown_test_acc = eval_result[6]
            all_test_acc = eval_result[7]
            test_unseen_mi = eval_result[8]
            known_test_mat = eval_result[9]
        else:
            pred = og_model.inference(data.x, data.edge_index)
            if (config.fold_type != "class_var"):
                known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat = eval_validation_model(pred,
                                                                                                               data.y,
                                                                                                               data.known_class_val_mask, 
                                                                                                               data.unknown_class_val_mask, 
                                                                                                               data.all_class_val_mask, 
                                                                                                               device)
            else:
                known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            
            known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi, known_test_mat = eval_test_model(pred, 
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
        "known_val_matched" : known_val_mat,
        "known_test_acc_run": known_test_acc,
        "unknown_test_acc_run": unknown_test_acc,
        "all_test_acc_run" : all_test_acc,
        "test_unseen_mi_run" : test_unseen_mi,
        "known_test_matched" : known_test_mat
    }
    return result_run_dict

      
    
    
def run_fixed_fold_experiment(config, log_all):

    gpu_number = config.gpu_number
    #device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    data = load_dataset(config.dataset, unknown_class_ratio=config.unknown_class_ratio, validation_split=True)
    og_model = setup_model(data, device, config, log_all=log_all)
    
    optimizer = torch.optim.Adam(og_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.sampler == "neighbor":
        og_model = train_with_sampler(og_model, optimizer, data, device, log_all, config)
    elif config.sampler == None:
        og_model = train(og_model, optimizer, data, device, log_all, config)
    else:
        raise NotImplementedError("Your sampler is not supported yet!")


    og_model.eval()
    data = data.to(device)
    pred = og_model.inference(data.x, data.edge_index)

    with torch.no_grad():
        
        final_known_val_acc, final_unknown_val_acc, final_all_acc, final_unseen_mi, final_known_val_mat = eval_validation_model(pred, 
                                                                                                                               data.y, 
                                                                                                                               data.known_class_val_mask, 
                                                                                                                               data.unknown_class_val_mask, 
                                                                                                                               data.all_class_val_mask,
                                                                                                                               device)

    return final_known_val_acc, final_unknown_val_acc, final_all_acc, final_unseen_mi, final_known_val_mat




    
    
        




