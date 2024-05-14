import random
import torch
import wandb
import numpy as np
from torch_geometric.nn.models import GCN
from models.DGI import DeepGraphInfomax, corrupt, readout
from evaluation import eval_validation_model
from pathlib import Path
from EarlyStopping import EarlyStopping
from tqdm import tqdm
from utils import labeled_val_fun
from models.kmeans import K_Means
from utils import cluster_acc, get_best_k
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def n_class_estimator(data, model, max_cand_k, split_ratio, num_val_cls, min_max_ratio, device):

    unlabeled_mask = data.all_class_test_mask | data.test_class_mask
    labeled_mask = ~unlabeled_mask

    print("Unlabled node classes: ", data.y[unlabeled_mask].unique())
    print("Labled node classes: ", data.y[labeled_mask].unique())
    
    u_num = data.y[unlabeled_mask].shape[0]
    
    print('extracting features for unlabeld data')
    u_targets = data.y[unlabeled_mask].detach().numpy()
    u_feats, _, _ = model(data.x, data.edge_index)
    u_feats = u_feats[unlabeled_mask, :].detach().numpy()
    
    cand_k = np.arange(max_cand_k)
    
    l_num = data.y[labeled_mask].shape[0]
    l_targets = data.y[labeled_mask].detach().numpy()
    l_feats, _, _ = model(data.x, data.edge_index)
    l_feats = l_feats[labeled_mask, :].detach().numpy()
    print('extracting features for labeld data')
    
    l_classes = data.y[data.all_class_val_mask].unique().tolist()
    num_lt_cls = int(round(len(l_classes)*split_ratio))
    lt_classes = set(random.sample(l_classes, num_lt_cls)) #random sample 5 classes from all labeled classes
    lv_classes = set(l_classes) - lt_classes
    
    print("Lt: ", lt_classes)
    print("Lv: ",lv_classes)
    
    lt_feats = np.empty((0, l_feats.shape[1]))
    lt_targets = np.empty(0)
    for c in lt_classes:
        lt_feats = np.vstack((lt_feats, l_feats[l_targets==c]))
        lt_targets = np.append(lt_targets, l_targets[l_targets==c])
    
    lv_feats = np.empty((0, l_feats.shape[1]))
    lv_targets = np.empty(0, dtype=np.int64)
    for c in lv_classes:
        lv_feats = np.vstack((lv_feats, l_feats[l_targets==c]))
        lv_targets = np.append(lv_targets, l_targets[l_targets==c])
    
    
    cvi_list = np.zeros(len(cand_k))
    acc_list = np.zeros(len(cand_k))
    cat_pred_list = np.zeros([len(cand_k),u_num+l_num])
    print('estimating K ...')
    for i in tqdm(range(len(cand_k))):
        
        cvi_list[i],  cat_pred_i = labeled_val_fun(np.concatenate((lv_feats, u_feats)), lt_feats, lt_targets, cand_k[i]+num_val_cls, device)
        cat_pred_list[i, :] = cat_pred_i
        acc_list[i] = cluster_acc(lv_targets, cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
        nmi_i = nmi_score(lv_targets, cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
        ari_i = ari_score(lv_targets, cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
        best_k = get_best_k(cvi_list[:i+1], acc_list[:i+1], cat_pred_list[:i+1], l_num, min_max_ratio=min_max_ratio)
        print("Tested k: ",cand_k[i]+num_val_cls)
        print('current best K {}, acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(best_k, acc_list[i], nmi_i, ari_i))
    
    kmeans = KMeans(n_clusters=best_k)
    u_pred = kmeans.fit_predict(u_feats).astype(np.int32) 
    acc, nmi, ari = cluster_acc(u_targets, u_pred), nmi_score(u_targets, u_pred), ari_score(u_targets, u_pred)
    print('Final K {}, acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(best_k, acc, nmi, ari))
    return best_k


def train_dgi(model, data, device, config):

    model = model.to(device)   
    model_path = Path("model_backup/" + config.model+"_"+str(config.dataset)+"_"
                      +str(config.num_protos)+"_"+str(config.fold_type)+"_n_class_estimate.pth")
    es = EarlyStopping(patience=config.patience, path=Path(model_path))

    optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config.lr, 
                                     weight_decay=config.weight_decay)
    
    model.train()
    for epoch in tqdm(range(config.n_epoch)):
        model.train()
            
        data = data.to(device)
        pos_z, neg_z, summary = model.forward(data.x, data.edge_index)   
        loss = model.loss(pos_z, neg_z, summary)
        wandb.log({'loss': loss})
        loss.backward()
        optimizer.step()


    return model
    
    
def estimate_number_of_classes(data, device, config):

    encoder = GCN(in_channels = data.x.shape[1],
                  hidden_channels = config.hidden_channels,
                  out_channels = None,
                  dropout = config.dropout,
                  num_layers =  config.num_layers)

    
    dgi = DeepGraphInfomax(hidden_channels = config.hidden_channels, 
                           encoder = encoder,
                           summary = readout,
                           corruption = corrupt)

    dgi = train_dgi(dgi, data, device, config)

    data = data.cpu()
    dgi = dgi.cpu()
    k_best = n_class_estimator(data, dgi, config.max_cand_k, config.split_ratio, config.num_val_cls, config.min_max_ratio, device)
    return k_best

    