import wandb
import torch

from torch_geometric.loader import NeighborLoader


from pathlib import Path
from tqdm import tqdm

from evaluation import eval_validation_model, eval_test_model
from EarlyStopping import EarlyStopping



def train(og_model, optimizer, data, device, log_all, config):
    
    og_model = og_model.to(device)   

    model_path = Path("model_backup/" + config.model+"_"+str(config.dataset)+"_"
                      +str(config.num_protos)+"_"+str(config.fold_type)+"_backup.pth")
    es = EarlyStopping(patience=config.patience, path=Path(model_path))
    
    og_model.train()
    for epoch in tqdm(range(config.n_epoch)):
        og_model.train()
            
        data = data.to(device)
        loss = og_model.train_one_epoch(optimizer, data)


        if (config.model != "openssl") & (config.fold_type != "class_var"):           
            og_model.eval()
            data = data.to(device)
            pred = og_model.inference(data.x, data.edge_index)
            known_val_acc, unknown_val_acc, all_acc, unseen_mi, known_val_mat = eval_validation_model(pred, 
                                                                                       data.y, 
                                                                                       data.known_class_val_mask, 
                                                                                       data.unknown_class_val_mask, 
                                                                                       data.all_class_val_mask,
                                                                                       device)
    
            if es(all_acc, og_model):
                og_model.load_state_dict(torch.load(model_path))
                break

    data = data.to(device)       
    og_model.post_train_processing(data)

    return og_model

def train_with_sampler(og_model, optimizer, data, device, log_all, config):
    og_model = og_model.to(device)
    data = data.to(device)

    model_path = Path("model_backup/" + config.model+"_"+str(config.dataset)+"_"
                      +str(config.num_protos)+"_"+str(config.fold_type)+"_backup.pth")
    es = EarlyStopping(patience=config.patience, path=Path(model_path))

    for epoch in tqdm(range(config.n_epoch)):
        og_model.train()

        known_class_val_mask_list = []
        unknown_class_val_mask_list  = []
        all_class_val_mask_list  = []
        pred_list = []
        y_list = []
        nl = NeighborLoader(data, 
                            num_neighbors=[config.batch_size[1], config.batch_size[2]], 
                            num_workers=4, 
                            batch_size=config.batch_size[0],
                            directed = True,
                            pin_memory=True)

        loss_agg = []
        for data_sampled in nl:
            loss = og_model.train_one_epoch(optimizer, data_sampled)
            loss_agg.append(float(loss))
          
            og_model.eval()
            data_sampled = data_sampled.to(device)

            if (config.model != "openssl") & (config.fold_type != "class_var"):                
                pred = og_model.inference(data_sampled.x, data_sampled.edge_index)
                pred_list.append(pred)
    
                y = data_sampled.y
                y_list.append(y)
    
                known_class_val_mask = data_sampled.known_class_val_mask
                known_class_val_mask_list.append(known_class_val_mask)
                unknown_class_val_mask = data_sampled.unknown_class_val_mask
                unknown_class_val_mask_list.append(unknown_class_val_mask)
                all_class_val_mask = data_sampled.all_class_val_mask
                all_class_val_mask_list.append(all_class_val_mask)        

        if (config.model != "openssl") & (config.fold_type != "class_var"):  
            with torch.no_grad():
                
                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)
        
                known_class_val_mask = torch.cat(known_class_val_mask_list, dim=0)
                unknown_class_val_mask = torch.cat(unknown_class_val_mask_list, dim=0)
                all_class_val_mask = torch.cat(all_class_val_mask_list, dim=0)
                
                known_val_acc, unknown_val_acc, all_acc, unseen_mi, known_val_mat = eval_validation_model(pred,
                                                                                           y,
                                                                                           known_class_val_mask, 
                                                                                           unknown_class_val_mask, 
                                                                                           all_class_val_mask,
                                                                                           device)
           
            if es(all_acc, og_model):
                og_model.load_state_dict(torch.load(model_path))
                break
            
          

    data = data.to(device)
    og_model.post_train_processing(data)

    return og_model


def eval_on_cpu(og_model, data, device, config):

    cpu_device = torch.device('cpu')

    data = data.to(cpu_device)
    og_model = og_model.to(cpu_device)
    og_model.device = cpu_device
    
    pred = og_model.inference(data.x, data.edge_index)
    
    if (config.fold_type != "class_var"):
        known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat = eval_validation_model(pred,
                                                                                                       data.y,
                                                                                                       data.known_class_val_mask, 
                                                                                                       data.unknown_class_val_mask, 
                                                                                                       data.all_class_val_mask, 
                                                                                                       device)
    else:
        known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi = (-1.0, -1.0, -1.0, -1.0)
        
    known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi, known_test_mat = eval_test_model(pred, 
                                                                                                     data.y, 
                                                                                                     data.known_class_test_mask, 
                                                                                                     data.unknown_class_test_mask, 
                                                                                                     data.all_class_test_mask,
                                                                                                     device)
    og_model.device = device

    return known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat, known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi, known_test_mat

    


def eval_with_sampler(og_model, data, device, config):
    
    data = data.to(device)
    nl = NeighborLoader(data, num_neighbors=[config.batch_size[1], config.batch_size[2]], 
                        num_workers=1,
                        batch_size=config.batch_size[0], 
                        pin_memory=True,
                        input_nodes = data.test_mask.cpu())
    
    pred_list = []
    y_list = []
    known_class_val_mask_list = []
    unknown_class_val_mask_list  = []
    all_class_val_mask_list  = []

    known_class_test_mask_list = []
    unknown_class_test_mask_list  = []
    all_class_test_mask_list  = []

    for data_sampled in nl:
        pred = og_model.inference(data_sampled.x, data_sampled.edge_index)
        pred_list.append(pred)
        
        y = data_sampled.y
        y_list.append(y)
        
        known_class_val_mask = data_sampled.known_class_val_mask
        known_class_val_mask_list.append(known_class_val_mask)
        unknown_class_val_mask = data_sampled.unknown_class_val_mask
        unknown_class_val_mask_list.append(unknown_class_val_mask)
        all_class_val_mask = data_sampled.all_class_val_mask
        all_class_val_mask_list.append(all_class_val_mask)

        known_class_test_mask = data_sampled.known_class_test_mask
        known_class_test_mask_list.append(known_class_test_mask)
        unknown_class_test_mask = data_sampled.unknown_class_test_mask
        unknown_class_test_mask_list.append(unknown_class_test_mask)
        all_class_test_mask = data_sampled.all_class_test_mask
        all_class_test_mask_list.append(all_class_test_mask)


    pred = torch.cat(pred_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    known_class_val_mask = torch.cat(known_class_val_mask_list, dim=0)
    unknown_class_val_mask = torch.cat(unknown_class_val_mask_list, dim=0)
    all_class_val_mask = torch.cat(all_class_val_mask_list, dim=0)

    known_class_test_mask = torch.cat(known_class_test_mask_list, dim=0)
    unknown_class_test_mask = torch.cat(unknown_class_test_mask_list, dim=0)
    all_class_test_mask = torch.cat(all_class_test_mask_list, dim=0)

    
        

    known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat = eval_validation_model(pred, y,
                                                                                       known_class_val_mask, 
                                                                                       unknown_class_val_mask, 
                                                                                       all_class_val_mask, device)
    
    known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi, known_test_mat = eval_test_model(pred, y,
                                                                                     known_class_test_mask, 
                                                                                     unknown_class_test_mask, 
                                                                                     all_class_test_mask, 
                                                                                     device)
    return known_val_acc, unknown_val_acc, all_val_acc, val_unseen_mi, known_val_mat, known_test_acc, unknown_test_acc, all_test_acc, test_unseen_mi, known_test_mat