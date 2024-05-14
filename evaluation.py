import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import cluster_acc
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score

def eval_validation_model(pred, y, known_class_val_mask, unknown_class_val_mask, all_class_val_mask, device):
    

    known_val_acc = accuracy_score(pred[known_class_val_mask].cpu().detach().numpy(),
                                   y[known_class_val_mask].cpu().detach().numpy())

    
    unknown_val_acc = cluster_acc(pred[unknown_class_val_mask].cpu().detach().numpy(),
                                     y[unknown_class_val_mask].cpu().detach().numpy())
    
    all_val_acc = cluster_acc(pred[all_class_val_mask].cpu().detach().numpy(),
                              y[all_class_val_mask].cpu().detach().numpy())
    
    unseen_nmi_val = adjusted_mutual_info_score(y[unknown_class_val_mask].cpu().detach().numpy(), 
                            pred[unknown_class_val_mask].cpu().detach().numpy())

    known_val_mat = cluster_acc(pred[known_class_val_mask].cpu().detach().numpy(),
                                     y[known_class_val_mask].cpu().detach().numpy())


    return known_val_acc, unknown_val_acc, all_val_acc, unseen_nmi_val, known_val_mat


def eval_test_model(pred, y, known_class_test_mask, unknown_class_test_mask, all_class_test_mask, device):
   
    known_test_acc = accuracy_score(pred[known_class_test_mask].cpu().detach().numpy(),
                                   y[known_class_test_mask].cpu().detach().numpy())


    if torch.sum(unknown_class_test_mask) != 0:
        unknown_test_acc = cluster_acc(pred[unknown_class_test_mask].cpu().detach().numpy(),
                                         y[unknown_class_test_mask].cpu().detach().numpy())
    else:
        unknown_test_acc = -1
    
    all_test_acc = cluster_acc(pred[all_class_test_mask].cpu().detach().numpy(),
                              y[all_class_test_mask].cpu().detach().numpy())
    
    unseen_nmi_test = adjusted_mutual_info_score(y[unknown_class_test_mask].cpu().detach().numpy(),
                            pred[unknown_class_test_mask].cpu().detach().numpy())

    known_test_mat = cluster_acc(pred[known_class_test_mask].cpu().detach().numpy(),
                                     y[known_class_test_mask].cpu().detach().numpy())

    return known_test_acc, unknown_test_acc, all_test_acc, unseen_nmi_test, known_test_mat