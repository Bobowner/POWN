import torch
import pandas as pd
import numpy as np
import math

from collections import defaultdict
from networkx import pagerank
from torch_geometric.utils import to_networkx
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from collections import Counter
from scipy.optimize import linear_sum_assignment
from models.kmeans import K_Means




def mean_and_std_tensor_list(np_list):
    np_list = np.asarray(np_list)
    
    mean = np_list.mean()
    std = np_list.std()

    return mean, std


def precompute_batch(num_neighbors=[128, 32], batch_size=32):

    loader = NeighborLoader(data, num_neighbors=num_neighbors, num_workers=4, batch_size=batch_size)
    
    subgraph_list = []
    
    for batch in loader:
        subgraph_list.append(batch)

    return subgraph_list


def reassign_labels(y, seen_labels, unseen_label_index):

    if isinstance(y, list):
        y = np.array(y)

    old_new_label_dict = {old_label:new_label for new_label, old_label in enumerate(seen_labels)}

    def convert_label(old_label):
        return old_new_label_dict[old_label] if old_label in old_new_label_dict else unseen_label_index

    new_y = [
        convert_label(label) for label in y
    ]

    new_y = np.array(new_y)

    return new_y


def reverse_labels(y, seen_labels, unseen_label_index):
    if isinstance(y, list):
        y = np.array(y)

    old_new_label_dict = {old_label:new_label for new_label, old_label in enumerate(seen_labels)}
    inverse_old_new_label_dict = {v: k for k, v in old_new_label_dict.items()}

    def convert_label(new_label):
        return inverse_old_new_label_dict[new_label] if new_label != unseen_label_index else unseen_label_index

    new_y = [convert_label(label) for label in y ]
    new_y = np.array(new_y)
    return new_y


def euclidean_distance(x, y):
    # Calculate element-wise squared differences
    squared_diff = (x - y)**2
    
    # Sum the squared differences along the feature dimension (axis=1)
    summed_squared_diff = squared_diff.sum(dim=1)
    
    # Take the square root to compute the Euclidean distance
    distance = summed_squared_diff.sqrt()
    
    return distance

def page_rank(data):
    nx_graph = to_networkx(data)
    scores = pagerank(nx_graph, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)

    return torch.Tensor(list(scores.values()))

  
def calc_avg_dicts(dict_list, blacklist=[]):
    df = pd.DataFrame(dict_list)
    average_dict = {}

    for key in df.columns:
        if key not in blacklist:
            average_dict[key + '_final'] = df[key].mean()

    return average_dict


def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T





def labeled_val_fun(u_feats, l_feats, l_targets, k, device):
    if device=='cuda':
        torch.cuda.empty_cache()
    l_num=len(l_targets)
    kmeans = K_Means(k, pairwise_batch_size=256)
    kmeans.fit_mix(torch.from_numpy(u_feats).to(device), torch.from_numpy(l_feats).to(device), torch.from_numpy(l_targets).to(device))
    cat_pred = kmeans.labels_.cpu().numpy() 
    u_pred = cat_pred[l_num:]
    silh_score = silhouette_score(u_feats, u_pred)
    return silh_score, cat_pred 


def get_best_k(cvi_list, acc_list, cat_pred_list, l_num, min_max_ratio):
    idx_cvi = np.max(np.argwhere(cvi_list==np.max(cvi_list)))
    idx_acc = np.max(np.argwhere(acc_list==np.max(acc_list)))
    idx_best = int(math.ceil((idx_cvi+idx_acc)*1.0/2))
    cat_pred = cat_pred_list[idx_best, :]
    cnt_cat = Counter(cat_pred.tolist())
    cnt_l = Counter(cat_pred[:l_num].tolist())
    cnt_ul = Counter(cat_pred[l_num:].tolist())
    bin_cat = [x[1] for x in sorted(cnt_cat.items())]
    bin_l = [x[1] for x in sorted(cnt_l.items())]
    bin_ul = [x[1] for x in sorted(cnt_ul.items())]
    best_k = np.sum(np.array(bin_ul)/np.max(bin_ul).astype(float)>=min_max_ratio)
    return best_k


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    
    assert y_pred.size == y_true.size
    # D = m_protos
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size