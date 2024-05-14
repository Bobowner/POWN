import torch
import numpy as np
import wandb
import torch.nn.functional as F

from torch_geometric.nn import LabelPropagation
from utils import gs
from models.pseudo_labels import pseudo_label_closest_proto, pseudo_label_lp, get_edge_weights


class OpenGraph(torch.nn.Module):
    def __init__(self, ssl, hidden_channels, num_protos, known_classes, unknown_classes, 
                 device, sup_loss_weight, pseudo_loss_weight, unsup_loss_weight, entropy_loss_weight,
                 geometric_loss_weight,
                 ood_percentile, lp_hop = 1, lp_min_seeds = 3, entropy_threshold = 0.1, sup_temp = 0.1, pseudo_temp = 0.7,
                 log_all=True, proto_type="param", pseudo_label_method = "none"):
        
        super().__init__()

        super(OpenGraph, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_protos = num_protos

        self.known_classes = known_classes.to(device)
        self.unknown_classes = unknown_classes.to(device)
        self.unlabeled_protos = [i for i in range(num_protos) if i not in known_classes]
        self.unlabeled_protos =  torch.tensor([i for i in range(num_protos) if i not in known_classes]).to(device)
        self.offset_array = self.calc_offset_array(torch.arange(num_protos), unknown_classes).to(device)

        self.sup_temp = sup_temp
        self.pseudo_temp = pseudo_temp
        self.lp_hop = lp_hop
        self.lp_min_seeds = lp_min_seeds
        self.entropy_threshold = entropy_threshold
        
        uniform_class_distr = torch.full((self.num_protos,), 1/self.num_protos, device=device)
        self.entropy_max = -torch.sum(uniform_class_distr * torch.log2(uniform_class_distr), dim=0).to(device)

        
        self.ood_percentile = ood_percentile
        self.pseudo_label_method = pseudo_label_method

        self.sup_loss_weight = sup_loss_weight
        self.pseudo_loss_weight = pseudo_loss_weight
        self.unsup_loss_weight = unsup_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.geometric_loss_weight = geometric_loss_weight

        
        self.ssl = ssl
        
        #self.encoder = encoder

        self.proto_type = proto_type

        self.device = device
        self.eps = 1e-9
        self.log_all = log_all
        


        self.prototypes = torch.nn.ModuleList()

        for i in range(num_protos):
            proto_i = ProtoRepre(hidden_channels, proto_type, device)
            self.prototypes.append(proto_i)

            #Gramm Schmidt
            #gs_v = gs(torch.stack(self.prototypes).numpy(), row_vecs=True, norm = True)
            #self.prototypes = [torch.tensor(gs_v[i], dtype=torch.float32).to(device) for i in range(num_protos)]


    def calc_offset_array(self, known_classes, unknown_classes):
        result = [torch.sum(elem > unknown_classes) for elem in known_classes]
        return torch.Tensor(result).type(torch.long)
            
    
    def forward(self, x, edge_index):
        features = self.ssl.encoder(x, edge_index)
       
        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)
        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
            
        probas = F.softmax(-distances, dim=1)
        return probas


    def inference(self, x, edge_index):
        
        features = self.ssl.encoder(x, edge_index)
       
        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device) # move init, check copy, extra method to return embeddin
        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
            
        probas = F.softmax(-distances, dim=1)

        return probas.argmax(dim=1)

    def get_embeddings(self, x, edge_index):
        features = self.ssl.encoder(x, edge_index)
        return features

    def update_proto(self, features, y, weights = None):
        if self.proto_type == "mean":
            for label in range(self.num_protos):
                self.prototypes[label].update_proto(features, weights, y,label)

    def post_train_processing(self, data):
        if self.proto_type == "mean":
            #TODO: Needs to be adjusted to running average for large graphs
            features = self.ssl.encoder(data.x, data.edge_index)
          
            n_mask, pseudo_labels = self.gen_pseudo_labels(features, data.edge_index, data)
            update_mask = data.labeled_mask | n_mask
            all_labels = torch.zeros(data.y.size(0), dtype=torch.long).to(self.device)
            all_labels[n_mask] = pseudo_labels[n_mask]
            all_labels[data.labeled_mask] = data.y[data.labeled_mask]
            weights = data.page_rank
            self.update_proto(features[update_mask], all_labels[update_mask], weights[update_mask])


    def train_one_epoch(self, optimizer, data):
        
             
        optimizer.zero_grad()

        features = self.ssl.encoder(data.x, data.edge_index)

        
        n_mask, pseudo_labels = self.gen_pseudo_labels(features, data.edge_index, data)

        pseudo_mask = data.unlabeled_mask & n_mask

        if self.proto_type == "mean":
            proto_update_mask = data.labeled_mask | n_mask 
            all_labels = torch.zeros(data.y.size(0), dtype=torch.long).to(self.device)
            all_labels[n_mask] = pseudo_labels[n_mask]
            all_labels[data.labeled_mask] = data.y[data.labeled_mask]
            
            weights = data.page_rank
            self.update_proto(features[proto_update_mask], all_labels[proto_update_mask], weights[proto_update_mask])


        open_loss = self.open_loss(features, pseudo_labels, pseudo_mask, data)
        open_loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

       
        

        return open_loss



    def open_loss(self, features, pseudo_labels, pseudo_mask, data):
        
        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)
        
        #unsupervised loss -> DGI
        unsup_loss = self.unsupervised_loss(data.x, data.edge_index, data.unlabeled_mask)

        #regularization to get an equal prototype distribution
        entropy_loss = self.entropy_regularizer(features, prototypes_tensor, data)

        #pseudo_label_loss
        if torch.sum(pseudo_mask) == 0:
            pseudo_sup_loss = 0
        elif self.proto_type == "mean":
            pseudo_sup_loss = self.supervised_loss(features[pseudo_mask], pseudo_labels[pseudo_mask], self.pseudo_temp)
        elif self.proto_type == "param":
            pseudo_sup_loss = self.proto_loss(features[pseudo_mask], 
                                       prototypes_tensor, 
                                       pseudo_labels[pseudo_mask],
                                       self.pseudo_temp)

        #supervised loss        
        if self.proto_type == "mean":
            sup_loss = self.supervised_loss(features[data.labeled_mask], data.y[data.labeled_mask], self.sup_temp)

        elif self.proto_type == "param":
            #sup_loss = self.supervised_loss(features[data.labeled_mask], data.y[data.labeled_mask], self.sup_temp)
            sup_loss = self.proto_loss(features[data.labeled_mask], 
                                       prototypes_tensor, 
                                       data.y[data.labeled_mask],
                                       self.sup_temp)

        if self.log_all:        
            wandb.log({'unsup_loss': unsup_loss, 'entropy_loss': entropy_loss, 'pseudo_loss': pseudo_sup_loss,'sup_loss': sup_loss})
        
        open_loss = self.sup_loss_weight*sup_loss + self.pseudo_loss_weight*pseudo_sup_loss + self.unsup_loss_weight*unsup_loss + self.entropy_loss_weight*entropy_loss
        open_loss += self.geometric_loss_weight*(self.class_seperability_loss(prototypes_tensor) + self.class_uniformity_loss(prototypes_tensor))
        return open_loss
        


    def proto_loss(self, features, prototypes_tensor, y, temperature):
        
        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
        distances = torch.div(distances, temperature)
        
        probabilities_for_training = torch.nn.Softmax(dim=1)(-distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), y]
        
        loss = -torch.log(probabilities_at_targets).mean()

        return loss
        

    def supervised_loss(self, features, target, temperature):
        cosine_dist = features @ features.t()
        cosine_mat = torch.div(cosine_dist, temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)
       
        cosine_mat = cosine_mat - mat_max.detach()

        target_ = target.contiguous().view(-1, 1)
        mask_pos = torch.eq(target_, target_.T)

        mask_neg_base = 1 - torch.diag(torch.ones(features.size(0))).to(self.device)

        pos_term = (cosine_mat * mask_pos).sum(1) / (mask_pos.sum(1) + self.eps)
        neg_term = (torch.exp(cosine_mat) * mask_neg_base).sum(1)
        
        log_term = (pos_term - torch.log(neg_term + self.eps))
        return -log_term.mean()
        
        
    def unsupervised_loss(self, x, edge_index, unlabeled_mask):
        
        pos_z, neg_z, summary = self.ssl.forward(x, edge_index, mask=unlabeled_mask)
        
        loss = self.ssl.loss(pos_z, neg_z, summary) 

        return loss

    def class_seperability_loss(self, prototypes_tensor):
        distances = torch.cdist(F.normalize(prototypes_tensor), F.normalize(prototypes_tensor), p=2.0)
        neg_max_dists = torch.max(-distances, dim=1).values
        loss = torch.exp(neg_max_dists)

        return torch.mean(loss)

    def class_uniformity_loss(self, prototypes_tensor):
        proto_center = torch.mean(prototypes_tensor, dim=0).unsqueeze(0)

        normalized_proto = F.normalize(prototypes_tensor - proto_center)

        cos_dist = normalized_proto @ normalized_proto.t()
        unit_matrix = torch.eye(cos_dist.shape[0]).to(self.device)
        cos_dist = cos_dist - unit_matrix
        
        loss = torch.max(cos_dist, 1).values
        
        return torch.mean(loss)


    def entropy_regularizer(self, features, prototypes_tensor, data):

        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)

        #Assumption: Label distribution of test/train is similar - to drop this assumption use uniform distr.
        labeled_classes, count_labels = torch.unique(data.y[data.labeled_mask], return_counts=True)
        n_remaining_protos = self.num_protos - labeled_classes.size(0)

        

        #set minus prorotypes, labeled classes
        combined = torch.cat((torch.arange(start=0, end=self.num_protos).to(self.device), labeled_classes))
        uniques, counts = combined.unique(return_counts=True)
        unlabeled_classes = uniques[counts == 1].type(torch.LongTensor)

        labeled_proto_ratio = labeled_classes.size(0) / self.num_protos

        prior = torch.zeros(self.num_protos).to(self.device)
        
        if labeled_classes.size(0) == 0:
            prior[unlabeled_classes] = (1/unlabeled_classes.size(0))*(1-labeled_proto_ratio)
        elif unlabeled_classes.size(0) == 0:
            prior[labeled_classes] = (count_labels/count_labels.sum(0))*labeled_proto_ratio
        else:
            prior[labeled_classes] = (count_labels/count_labels.sum(0))*labeled_proto_ratio
            prior[unlabeled_classes] = (1/unlabeled_classes.size(0))*(1-labeled_proto_ratio)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        probas = F.log_softmax(-distances, dim=1).max()

        loss =  kl_loss(probas, prior)

        return distances.mean()

    def select_threshold(self, closest, percentile):
        q = torch.quantile(closest, percentile)
        return q


    def everything_to_device(self, device):
        self = self.to(device)        
        self.known_classes = self.known_classes.to(device)
        self.unknown_classes = self.unknown_classes.to(device)
        self.unlabeled_protos =  self.unlabeled_protos.to(device)
        self.offset_array = self.offset_array.to(device)
        self.entropy_max = self.entropy_max.to(device)
        self.device = device
        return self
        

    def gen_pseudo_labels(self, features, edge_index, data):
        #pseudo label all data points as id that are close to a known class prototype
        
        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)

        if len(self.unlabeled_protos) == 0:
            n_mask = torch.zeros_like(data.y)
            pseudo_labels = torch.zeros_like(data.y)

        elif self.pseudo_label_method == "closest":

            n_mask, pseudo_labels =  pseudo_label_closest_proto(features=features, 
                                                                edge_index=edge_index, 
                                                                prototypes_tensor=prototypes_tensor, 
                                                                proto_subset = self.unlabeled_protos,  
                                                                known_classes=self.known_classes, 
                                                                ood_percentile=self.ood_percentile, 
                                                                data=data)
        elif self.pseudo_label_method == "lp":
            n_mask, pseudo_labels =  pseudo_label_lp(features = features, 
                                                     edge_index = edge_index, 
                                                     prototypes_tensor = prototypes_tensor, 
                                                     num_protos = self.num_protos, 
                                                     proto_subset = self.unlabeled_protos,
                                                     known_classes = self.known_classes,
                                                     ood_percentile = self.ood_percentile, 
                                                     lp_hop = self.lp_hop, 
                                                     entropy_threshold = self.entropy_threshold,
                                                     entropy_max = self.entropy_max,
                                                     eps = self.eps, 
                                                     data = data, 
                                                     device = self.device)

            

        return n_mask, pseudo_labels


    def get_edge_weights(self, data):
        prototypes_tensor = torch.stack([proto() for proto in self.prototypes]).to(self.device)
        features = self.ssl.encoder(data.x, data.edge_index)
                
        return get_edge_weights(features, data.edge_index, prototypes_tensor, self.eps)
    
# ToDO: Neighbor versus random batch versus full batch
    def sampler(self, data, n_nodes, batch_size):
        indices = torch.arange(0,n_nodes)
        perm = torch.randperm(n_nodes)
        sample = perm[:batch_size]

        sampled_data = data.subgraph(sample)

        return sampled_data
        


class ProtoRepre(torch.nn.Module):
    def __init__(self, hidden_channels, proto_type, device):
        
        super(ProtoRepre, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.proto_type = proto_type
        self.normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

        self.device = device

        
        if proto_type == "param":
            self.prototype = torch.Tensor(hidden_channels)
            torch.nn.init.normal_(self.prototype, mean=0.0, std=1.0)
            self.prototype = torch.nn.functional.normalize(self.prototype, p=2.0, dim=0)
            self.prototype = torch.nn.Parameter(self.prototype).to(device)

        elif proto_type == "mean":
            self.prototype = torch.Tensor(hidden_channels)
            torch.nn.init.normal_(self.prototype, mean=0.0, std=1.0)
            self.prototype = torch.nn.functional.normalize(self.prototype, p=2.0, dim=0).to(device)
            #self.prototype = torch.nn.Parameter(self.prototype, requires_grad=False)

    def update_proto(self, features, weights, y, label):
        
        if self.proto_type == "mean":

            class_wise_features = features[y==label,:]
            weights = F.softmax(weights[y==label], dim=0).unsqueeze(1).expand_as(class_wise_features)
            #res = torch.matmul(class_wise_features, weights)
            self.prototypes = self.normalizer(torch.sum(class_wise_features*weights, dim=0).to(self.device))
        else:
            raise NotImplementedError("Your prototype update is not defined!")
        
        

        # self.gat_layer = GATConv(in_channels=hidden_state, out_channels=hidden_state, heads=1, concat=False) #paper variant?
        #self.atten = nn.MultiheadAttention(hidden_state, 2) #code variant
    
    def forward(self):#, spt_embedding_i, degree_list_i=None):

        if self.proto_type != None:
            
            return self.prototype
            
        else:
            
            if degree_list_i == None:
                
                avg_proto_i = torch.sum(spt_embedding_i, 0) / spt_embedding_i.shape[0]
                attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
            
            else:
                
                norm_degree = degree_list_i / torch.sum(degree_list_i)
                norm_degree = norm_degree.unsqueeze(1)
                avg_proto_i = torch.sum(torch.mul(spt_embedding_i, norm_degree), 0)
                attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
                
            attn_output, attn_output_weights = self.atten(attn_input, attn_input, attn_input)
            proto_embedding_i = attn_output[0] + avg_proto_i.unsqueeze(0)
            return proto_embedding_i, attn_output_weights
