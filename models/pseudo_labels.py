import torch
import torch.nn.functional as F
from torch_geometric.nn import LabelPropagation
from utils import euclidean_distance


def select_threshold(closest, percentile):
    q = torch.quantile(closest, percentile)
    return q

def compute_novel_mask(features, edge_index, prototypes_tensor, known_classes, ood_percentile, data):
    #seelct a threshold such that ood_percentile parts of the labeled data are in 
    #And compute mask for novel classes based on this threshold
    
    cosine_sim = F.normalize(features) @ F.normalize(prototypes_tensor).t()

    if len(known_classes) == 0:
        n_mask = torch.ones_like(data.y, dtype=torch.bool)
        return n_mask
    balanced_sim = (cosine_sim[:,known_classes].clone() 
                    - cosine_sim[:,known_classes].mean(dim=1, keepdims=True)) / cosine_sim[: , known_classes].std(dim=1, keepdims=True)

    balanced_sim_max, _ = torch.max(balanced_sim, dim=1)

    if balanced_sim_max[data.labeled_mask].numel() == 0:
        ood_threshold = 0
    else:
        ood_threshold = select_threshold(balanced_sim_max[data.labeled_mask], ood_percentile)
    n_mask = balanced_sim_max < ood_threshold

    return n_mask

def pseudo_label_closest_proto(features, edge_index, prototypes_tensor, proto_subset, known_classes, ood_percentile, data):
   
    n_mask = compute_novel_mask(features=features, 
                                edge_index=edge_index, 
                                prototypes_tensor=prototypes_tensor, 
                                known_classes=known_classes, 
                                ood_percentile=ood_percentile, 
                                data=data)

    cosine_sim = F.normalize(features) @ F.normalize(prototypes_tensor).t()

    balanced_sim = (cosine_sim[:, proto_subset].clone() 
                    - cosine_sim[:,proto_subset].mean(dim=1, keepdims=True)) / cosine_sim[:, proto_subset].std(dim=1, keepdims=True)

    closest_indices_in_subset = torch.argmax(balanced_sim, dim=1)
    
    closest_indices = torch.index_select(proto_subset, 0, closest_indices_in_subset)
                
    pseudo_labels = closest_indices

    return n_mask, pseudo_labels


def pseudo_label_lp_step(features, prototypes_tensor, edge_weights, lp_mask, proto_subset, data, num_protos, lp_hop, ood_percentile, device):
    #Init label propagation
    lp = LabelPropagation(num_layers=lp_hop, alpha=0.9)
    pseudo_labels = data.y.detach().clone()
    pseudo_label_mask = torch.zeros_like(data.y, dtype=torch.bool)


    #Determine seed points for label propagation
    cosine_sim = F.normalize(features) @ F.normalize(prototypes_tensor).t()
    balanced_sim = (cosine_sim[:,proto_subset].clone() 
                    - cosine_sim[:,proto_subset].mean(dim=1, keepdims=True)) / cosine_sim[:,proto_subset].std(dim=1, keepdims=True)

    closest_indices_in_subset = torch.argmax(balanced_sim, dim=1)
    #map subset back to proto indices
    closest_indices = torch.index_select(proto_subset, 0, closest_indices_in_subset)


    #count occurences of pseudo labels, determine ratio to select
    unique_labels_n, counts_labels_n = torch.unique(closest_indices, return_counts=True)
    counts = torch.zeros(num_protos, dtype=torch.long).to(device)
    counts[unique_labels_n] = counts_labels_n
    counts = counts*ood_percentile

    

    for p in range(proto_subset.size(0)):
        #select the closest k datapoints per prototype and use them as seed pseudolabels
        k =  max(0, int(counts[proto_subset[p]]))
        values, indices = torch.topk(balanced_sim[:,p], k, dim=0, largest=True)

        #make sure to only use labels where dp is closest to prototype as well
        set_indices = indices[closest_indices[indices] == p]
        pseudo_labels[set_indices] = proto_subset[p]
        pseudo_label_mask[set_indices] = True

    #mask: unlabeled, proablby novel classes and got a pseudo label
    #pseudo_labels = closest_indices

    #performing lp
    lp_mask = pseudo_label_mask & lp_mask
    pseudo_probs = lp(pseudo_labels, data.edge_index, lp_mask, edge_weight=edge_weights)

    #Only keep labels with low uncertainty
    temperature_lp = 10
    
    pseudo_probs = F.softmax(pseudo_probs*temperature_lp, dim = 1)
    proto_probs = F.softmax(cosine_sim[:,proto_subset]*temperature_lp, dim = 1)

    return pseudo_probs, proto_probs


def get_edge_weights(features, edge_index, prototypes_tensor, eps):
    #compute edge weights based 
    src = edge_index[0,:]
    dst = edge_index[1,:]
    edge_weights_nodes = euclidean_distance(F.normalize(features[src,:]), F.normalize(features[dst,:]))
    edge_proto_dist = torch.cdist(F.normalize(features)[src,:], F.normalize(prototypes_tensor), p=2.0)
    
    edge_weight_proto = edge_proto_dist.min(dim=1).values
    edge_weights = 1/( edge_weight_proto + edge_weights_nodes + eps) # edge_weights_nodes +

    return edge_weights


def pseudo_label_lp(features, edge_index, prototypes_tensor, num_protos, proto_subset, 
                    known_classes, ood_percentile, lp_hop, entropy_threshold, entropy_max, eps, data, device):
    #compute novel mask
    n_mask = compute_novel_mask(features=features, 
                                edge_index=edge_index, 
                                prototypes_tensor=prototypes_tensor, 
                                known_classes=known_classes, 
                                ood_percentile=ood_percentile, 
                                data=data)

    #compute edge weights based 
    #src = data.edge_index[0,:]
    #dst = data.edge_index[1,:]
    #edge_weights_nodes = euclidean_distance(F.normalize(features[src,:]), F.normalize(features[dst,:]))
    #edge_proto_dist = torch.cdist(F.normalize(features)[src,:], F.normalize(prototypes_tensor), p=2.0)
    
    #edge_weight_proto = edge_proto_dist.min(dim=1).values
    #edge_weights = 1/( edge_weight_proto + eps) # edge_weights_nodes +

    edge_weights = get_edge_weights(features, edge_index, prototypes_tensor, eps)


    #pseudo labels for novel samples
    lp_mask =  n_mask & data.unlabeled_mask
    pseudo_probs, proto_probs = pseudo_label_lp_step(features, prototypes_tensor, edge_weights, lp_mask, proto_subset, data, num_protos, lp_hop, ood_percentile, device)
    
    entropies_lp = -torch.sum(pseudo_probs * torch.log2(pseudo_probs), dim=1) / entropy_max
    entropies_proto = -torch.sum(proto_probs * torch.log2(proto_probs), dim=1) / entropy_max
    entropy_weight = 0.2
    entropies = entropy_weight*entropies_lp + (1-entropy_weight)*entropies_proto
    
    if entropies[n_mask & data.unlabeled_mask].numel() == 0:
        entopy_perc = 0
    else:
        entopy_perc = select_threshold(entropies[n_mask & data.unlabeled_mask], entropy_threshold)

    pseudo_labels = pseudo_probs.max(dim=1).indices
    n_mask = n_mask & (entropies < entopy_perc)

    return n_mask, pseudo_labels