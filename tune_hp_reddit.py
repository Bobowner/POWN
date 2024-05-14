import torch
import torch.nn.functional as F
import torch_geometric
import matplotlib.pyplot as plt
import wandb
import argparse


from torch_geometric.nn.models import GraphSAGE, GCN, GAT
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Reddit, Reddit2, Planetoid
from functools import partial

from handle_meta_data import load_yml, built_config_with_default
from open_dataset import load_reddit2
from EarlyStopping import EarlyStopping
#from evaluation import eval_validation_model


def train(og_model, optimizer, data, device, log_all):
    
    og_model = og_model.to(device)
    data = data.to(device)
    #data_sampled = data

    model_path = Path("model_backup/tests_reddit_gcn_backup.pth")
    es = EarlyStopping(patience=30, path=Path(model_path))
    

    loss_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    
    for epoch in tqdm(range(300)):
        og_model.train()
        data = data.to(device)

        nl = NeighborLoader(data, 
                            num_neighbors=[64, 16], 
                            num_workers=4, 
                            batch_size=128, 
                            pin_memory=True,
                            input_nodes = data.train_mask.cpu())


        loss_epoch = []
        for data_sampled in nl:
            optimizer.zero_grad()
            og_model.train()
            
            logits = og_model(data_sampled.x, data_sampled.edge_index)
            out = F.softmax(logits, dim=1)
            loss = F.cross_entropy(out[data_sampled.train_mask], data_sampled.y[data_sampled.train_mask])
            
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss)
            
        loss_list.append(torch.Tensor(loss_epoch).mean())
    

        with torch.no_grad():
            og_model.eval()
            og_model.cpu()
            data = data.cpu()
            pred = og_model(data.x, data.edge_index).argmax(dim=1)

            train_acc = accuracy_score(pred[data.train_mask].cpu().numpy(), data.y[data.train_mask].cpu().numpy())
            train_acc_list.append(train_acc)
            
            val_acc = accuracy_score(pred[data.val_mask].cpu().numpy(), data.y[data.val_mask].cpu().numpy())
            val_acc_list.append(val_acc)
            
            test_acc = accuracy_score(pred[data.test_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy())
            test_acc_list.append(test_acc)

            og_model.to(device)
            data = data.to(device)


        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": torch.Tensor(loss_epoch).mean(),
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        )
            
        

        if es(val_acc, og_model):
            og_model.load_state_dict(torch.load(model_path))
            break

    return loss_list, train_acc_list, val_acc_list, test_acc_list

def main():

    #dataset = Reddit2(root="dataset")
    #dataset = Planetoid(root='/tmp/Cora', name='Cora')
    wandb.init(project=project_name, mode="online")

    data = load_reddit2()
    device = torch.device(*('cuda', 1) if torch.cuda.is_available() else 'cpu')
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    hidden_channels = wandb.config.hidden_channels
    num_layers = 2 #wandb.config.num_layers
    dropout = wandb.config.dropout
    log_all = True
    
    
    
    model = GCN(in_channels = data.x.shape[1],
                     hidden_channels = hidden_channels,
                     out_channels = int(data.y.max())+1,
                     dropout = dropout,
                     #norm = "layer_norm",
                     num_layers = num_layers)
                     #kwars = {"heads" : 8})

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr= lr, 
                                 weight_decay=weight_decay)
    
    loss, train_acc, val_acc, testacc= train(model, optimizer, data, device, log_all)
    final_val_acc = val_acc[-1]
    wandb.log({"final_val_acc":final_val_acc})

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',nargs='+', default=None, help="Pass name of the dataset you use")
ARGS = parser.parse_args()
project_name = "tune_gcn_"+dataset

path = Path("experiments/tune_hp_gcn_reddit.yml")
sweep_configuration = load_yml(path)
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
wandb.agent(sweep_id, function=main, count=100)