import torch
import numpy as np
import copy
from torch_geometric.datasets import Planetoid, Amazon, Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
from utils import page_rank
from pathlib import Path


def load_arxiv():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    data.node_year = data.node_year.squeeze()
    data.y = data.y.squeeze()
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    num_nodes = data.num_nodes

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = 1

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_idx] = 1

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = 1
    
    return data

def load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0):
    dataset = Planetoid(root='dataset/' + name, name=name)
    data = dataset[0]#.to(device)

    #train_mask, val_mask, test_mask = create_split(data, train_portion, val_portion, test_portion, seed)
    
    #data.train_mask = train_mask
    #data.val_mask = val_mask
    #data.test_mask = test_mask

    return data

def load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0):
    dataset = Amazon(root='dataset/' + "amazon_" + str(name), name=name)
    data = dataset[0]

    #train_mask, val_mask, test_mask = create_split(data, train_portion, val_portion, test_portion, seed)
    splits = torch.load("dataset/"+name+"/own_23_splits.pt")

    data.train_mask = splits["train"][:,0]
    data.val_mask = splits["valid"][:,0]
    data.test_mask = splits["test"][:,0]

    return data

def load_reddit2():
    dataset = Reddit2(root="dataset")
    data = dataset[0]
    #data.edge_index = to_undirected(data.edge_index)
    return data


def create_class_folds(data, unknown_class_ratio):
    classes = torch.unique(data.y)
    n_classes = len(classes)
    
    # Generate a random permutation of indices
    indices = torch.randperm(n_classes)
    
    # Use the indices to shuffle the tensor
    classes = classes[indices]
    fold_length =int(max(unknown_class_ratio * n_classes, 2))
    
    # Split the tensor into equal-sized folds using a loop
    folds = [classes[i:i+fold_length] for i in range(0, n_classes, fold_length)]

    if len(folds[-1])==1:
        combined_fold = torch.cat((folds[-2], folds[-1]))
        # Replace the last two folds with the combined fold
        folds = folds[:-2]
        folds.append(combined_fold)

    return folds


def prepare_fold_masks(data, folds, val_fold_index, test_fold_index):

    data = copy.deepcopy(data)
    data.classes = torch.unique(data.y)
    known_classes = torch.cat([folds[i] for i in range(len(folds)) if i != val_fold_index and i != test_fold_index]).sort().values
    val_classes = folds[val_fold_index].sort().values
    test_classes = folds[test_fold_index].sort().values

    data.known_classes = known_classes
    data.val_classes = val_classes
    data.test_classes = test_classes
    data.unknown_classes = torch.cat((val_classes,test_classes))
    
    #train mask
    known_class_mask = torch.isin(data.y, known_classes)
    data.known_class_mask = known_class_mask
    data.labeled_mask = known_class_mask & data.train_mask

    #val mask
    data.val_class_mask = torch.isin(data.y, val_classes)
    data.known_class_val_mask = known_class_mask & data.val_mask
    data.unknown_class_val_mask = data.val_class_mask & data.val_mask
    data.all_class_val_mask = (known_class_mask | data.val_class_mask) & data.val_mask

    #test class mask
    test_class_mask = torch.isin(data.y, test_classes)
    data.test_class_mask = test_class_mask
    data.known_class_test_mask = known_class_mask & data.test_mask
    data.unknown_class_test_mask = test_class_mask & data.test_mask
    data.all_class_test_mask = (known_class_mask | data.test_class_mask) & data.test_mask

    
    data.unlabeled_mask = ~data.labeled_mask

    return data

def prepare_fold_class_variation(data, folds, train_test_index):

    data = copy.deepcopy(data)
    data.classes = torch.unique(data.y)
    
    known_classes = [folds[i] for i in range(len(folds)) if i < train_test_index]
    if known_classes == []:
        known_classes = torch.empty((0), dtype=torch.float32)
    else:
        known_classes = torch.cat(known_classes).sort().values
    val_classes = torch.tensor([])#folds[val_fold_index].sort().values

    
    test_classes = [folds[i] for i in range(len(folds)) if i >= train_test_index]
    if test_classes == []:
        test_classes = torch.empty((0), dtype=torch.float32)
    else:
        test_classes = torch.cat(test_classes).sort().values

    data.known_classes = known_classes
    data.val_classes = val_classes
    data.test_classes = test_classes
    data.unknown_classes = torch.cat((val_classes,test_classes))
    
    #train mask
    known_class_mask = torch.isin(data.y, known_classes)
    data.known_class_mask = known_class_mask
    data.labeled_mask = known_class_mask & data.train_mask

    #val mask
    data.val_class_mask = torch.isin(data.y, val_classes)
    data.known_class_val_mask = known_class_mask & data.val_mask
    data.unknown_class_val_mask = data.val_class_mask & data.val_mask
    data.all_class_val_mask = (known_class_mask | data.val_class_mask) & data.val_mask

    #test class mask
    test_class_mask = torch.isin(data.y, test_classes)
    data.test_class_mask = test_class_mask
    data.known_class_test_mask = known_class_mask & data.test_mask
    data.unknown_class_test_mask = test_class_mask & data.test_mask
    data.all_class_test_mask = (known_class_mask | data.test_class_mask) & data.test_mask
    
    data.unlabeled_mask = ~data.labeled_mask

    return data


def create_fold_data(data, name, unknown_class_ratio):

    path = Path("fold_indices/"+name+"_class_split_"+str(unknown_class_ratio)+".pt")
    
    if path.is_file():
        folds = torch.load(path)
    else:
        folds = create_class_folds(data, unknown_class_ratio)
        torch.save(folds, path)
    
    n_folds = len(folds)
    datasets = []
    
    for test_fold_idx in range(n_folds):
        val_fold_idx = (test_fold_idx - 1) % n_folds
        data_new = prepare_fold_masks(data, folds, val_fold_idx, test_fold_idx)
        datasets.append(data_new)

    return datasets

def create_fold_data_class_variation(data, name, unknown_class_ratio):

    path = Path("fold_indices/"+name+"_class_variation_class_split_"+str(unknown_class_ratio)+".pt")
    
    if path.is_file():
        folds = torch.load(path)
    else:
        folds = create_class_folds(data, unknown_class_ratio)
        torch.save(folds, path)
    
    n_folds = len(folds)
    datasets = []
    
    for test_split in range(n_folds+1):
        data_new = prepare_fold_class_variation(data, folds, test_split)
        datasets.append(data_new)

    datasets = reversed(datasets)
    return datasets


def create_fold_data_resampling(data, name, unknown_class_ratio, n_folds):

    path = Path("fold_indices/"+name+"_class_split_"+str(unknown_class_ratio)+"_w_resampling"+".pt")
    
    if path.is_file():
        folds = torch.load(path)
    else:
        folds = folds = create_folds_with_resampling(data, unknown_class_ratio, n_folds)
        torch.save(folds, path)
    
    n_folds = len(folds)
    datasets = []
    
    for fold in folds:
        data_new = prepare_fold_masks_resampling(data, fold)
        datasets.append(data_new)

    return datasets


def create_folds_with_resampling(data, unknown_class_ratio, n_folds):
    folds = []
    
    for i in range(n_folds):
        classes = data.y.unique()
        n_classes = classes.max()+1
        indices = torch.randperm(n_classes)
        classes = classes[indices]
        n_test_classes = max(int(unknown_class_ratio * n_classes), 2)
        test_classes = classes[:n_test_classes]
        classes = classes[n_test_classes:]
        n_val_classes = max(int(unknown_class_ratio * classes.shape[0]), 2)
        val_classes = classes[:n_val_classes]
        train_classes = classes[n_val_classes:]
        folds.append((train_classes, val_classes, test_classes))
    return folds

def prepare_fold_masks_resampling(data, fold):
    data.classes = torch.unique(data.y)
    data.known_classes = fold[0].sort().values
    data.val_classes = fold[1].sort().values
    data.test_classes = fold[2].sort().values
    data.unknown_classes = torch.cat((data.val_classes,data.test_classes))

    #train mask
    known_class_mask = torch.isin(data.y, data.known_classes)
    data.known_class_mask = known_class_mask
    data.labeled_mask = known_class_mask & data.train_mask

    #val mask
    data.val_class_mask = torch.isin(data.y, data.val_classes)
    data.known_class_val_mask = known_class_mask & data.val_mask
    data.unknown_class_val_mask = data.val_class_mask & data.val_mask
    data.all_class_val_mask = (known_class_mask | data.val_class_mask) & data.val_mask

    #test class mask
    test_class_mask = torch.isin(data.y, data.test_classes)
    data.test_class_mask = test_class_mask
    data.known_class_test_mask = known_class_mask & data.test_mask
    data.unknown_class_test_mask = test_class_mask & data.test_mask
    data.all_class_test_mask = (known_class_mask | data.test_class_mask) & data.test_mask

    
    data.unlabeled_mask = ~data.labeled_mask

    return data


def create_class_split(data, unknown_class_ratio, validation_split, fixed):
    # split class ratio of class for unknown
    classes = torch.unique(data.y)
    n_classes = classes.shape[0]
    n_unknown_classes = int(n_classes*unknown_class_ratio)
    
    n_known_classes = n_classes - n_unknown_classes
    if fixed:
        indices = torch.arange(n_classes)
    else:
        indices = torch.randperm(n_classes)
    
    known_classes = classes[indices[:n_known_classes]]
    unknown_classes = classes[indices[-n_unknown_classes:]]

    
    data.known_classes = known_classes.sort().values
    data.test_classes = unknown_classes.sort().values

    #data.known_classes = known_classes
    #data.val_classes = val_classes
    #data.test_classes = test_classes
    data.unknown_classes = data.test_classes#torch.cat((val_classes,test_classes))


    if validation_split:

        #split class ratio of known classes
        n_val_classes = int(n_known_classes*unknown_class_ratio)
     

        if fixed:
            indices = torch.arange(n_known_classes)
        else:
            indices = torch.randperm(n_known_classes)

        n_known_classes = n_known_classes - n_val_classes


        val_classes = known_classes[indices[-n_val_classes:]]
        known_classes = known_classes[indices[:n_known_classes]]        

        data.known_classes = known_classes.sort().values
        data.val_classes = val_classes.sort().values
        data.unknown_classes = torch.cat([data.unknown_classes , data.val_classes])

    
    data.classes = classes.sort().values

    return data


def create_masks(data, classes, known_classes, unknown_classes, validation_split):

    
    known_class_mask = torch.isin(data.y, known_classes)
    unknown_class_mask = ~known_class_mask

    data.known_class_mask = known_class_mask
    data.test_class_mask = torch.isin(data.y, data.test_classes)
    data.unknown_class_mask = unknown_class_mask

    if validation_split:
        data.val_class_mask = torch.isin(data.y, data.val_classes)
        data.unknown_class_val_mask = data.val_class_mask & data.val_mask
        data.known_class_val_mask = known_class_mask & data.val_mask
        data.all_class_val_mask = (known_class_mask | data.val_class_mask) & data.val_mask
        data.known_class_test_mask = known_class_mask & data.test_mask
        data.unknown_class_test_mask = unknown_class_mask & data.test_mask
        data.all_class_test_mask = (data.known_class_mask | data.test_class_mask) & data.test_mask


    else:    
        data.known_class_test_mask = known_class_mask & data.test_mask
        data.unknown_class_test_mask = unknown_class_mask & data.test_mask

    data.labeled_mask = known_class_mask & data.train_mask
    data.unlabeled_mask = ~data.labeled_mask

    return data

def load_dataset(name, unknown_class_ratio=0.2, validation_split=False, fixed=False):
    if name == "ogb-arxiv":
        data = load_arxiv()
    elif (name == "cora") | (name == "citeseer"):
        data = load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif (name == "photo") | (name == "computers"):
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "reddit2":
        data = load_reddit2()
    else:
      raise NotImplementedError("Your choosen dataset: "+name+" is not supported")


    #page rank scores
    path = Path("dataset/" + name+"/"+"page_rank_scores.pt")
    if path.is_file():
        data.page_rank = torch.load(path)
    else:
        pr_scores = page_rank(data)
        data.page_rank = pr_scores
        torch.save(pr_scores, path)

    data = create_class_split(data, unknown_class_ratio, validation_split, fixed)
    data = create_masks(data, data.classes, data.known_classes, data.unknown_classes, validation_split)
    return data

def load_folds(name, unknown_class_ratio=0.2, fixed=False):
    if name == "ogb-arxiv":
        data = load_arxiv()
    elif (name == "cora") | (name == "citeseer"):
        data = load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "photo":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "computers":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "reddit2":
        data = load_reddit2()
    else:
      raise NotImplementedError("Your choosen dataset: "+name+" is not supported")


    #page rank scores
    path = Path("dataset/" + name+"/"+"page_rank_scores.pt")
    if path.is_file():
        data.page_rank = torch.load(path)
    else:
        pr_scores = page_rank(data)
        data.page_rank = pr_scores
        torch.save(pr_scores, path)

    datasets = create_fold_data(data, name, unknown_class_ratio)

    return datasets

def load_folds_class_variation(name, unknown_class_ratio=0.2, fixed=False):
    if name == "ogb-arxiv":
        data = load_arxiv()
    elif (name == "cora") | (name == "citeseer"):
        data = load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "photo":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "computers":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "reddit2":
        data = load_reddit2()
    else:
      raise NotImplementedError("Your choosen dataset: "+name+" is not supported")


    #page rank scores
    path = Path("dataset/" + name+"/"+"page_rank_scores.pt")
    if path.is_file():
        data.page_rank = torch.load(path)
    else:
        pr_scores = page_rank(data)
        data.page_rank = pr_scores
        torch.save(pr_scores, path)

    datasets = create_fold_data_class_variation(data, name, unknown_class_ratio)

    return datasets

def load_folds_resampling(name, unknown_class_ratio=0.2, n_folds=5, fixed=False):
    if name == "ogb-arxiv":
        data = load_arxiv()
    elif (name == "cora") | (name == "citeseer"):
        data = load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "photo":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "computers":
        data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
    elif name == "reddit2":
        data = load_reddit2()
    else:
      raise NotImplementedError("Your choosen dataset: "+name+" is not supported")


    #page rank scores
    path = Path("dataset/" + name+"/"+"page_rank_scores.pt")
    if path.is_file():
        data.page_rank = torch.load(path)
    else:
        pr_scores = page_rank(data)
        data.page_rank = pr_scores
        torch.save(pr_scores, path)

    datasets = create_fold_data_resampling(data, name, unknown_class_ratio, n_folds)

    return datasets


def create_split(data, train_portion, val_portion, test_portion, seed=None):
    
    y = data.y.cpu().detach().numpy()
    unique, counts = np.unique(y, return_counts=True)

    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []

    for cl in unique:
        
        tmp = np.argwhere(y==cl)
        c1 = int(len(tmp)*train_portion)
        c2 = int(len(tmp)*(train_portion+val_portion))
        rng.shuffle(tmp)
        train.append(tmp[:c1])
        val.append(tmp[c1:c2])
        test.append(tmp[c2:])
        
    train_ix = np.concatenate(train)
    val_ix = np.concatenate(val)
    test_ix = np.concatenate(test)

    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    train_mask[train_ix] = True
    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[val_ix] = True
    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[test_ix] = True
    return train_mask, val_mask, test_mask
