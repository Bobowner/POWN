class OrcaGraph(torch.nn.Module):
    def __init__(self, 
                 encoder, 
                 hidden_channels, 
                 num_protos, 
                 known_classes, 
                 unknown_classes, 
                 device,
                 ood_percentile, 
                 lp_hop = 2, 
                 lp_min_seeds = 3, 
                 entropy_threshold = 0.1,
                 log_all=True, 
                 proto_type="param", 
                 pseudo_label_method = "none"):

         self.encoder = encoder
         self.hidden_channels = hidden_channels
         self.num_protos = num_protos
         self.known_classes = known_classes
         self.unknown_classes = unknown_classes
         self.device = device
         self.ood_percentile = ood_percentile

        self.parinwise_temperature = 0.1



     def train_one_epoch(self, optimizer, data):
        
             
        optimizer.zero_grad()

        features = self.encoder(data.x, data.edge_index)

        pseudo_mask = data.unlabeled_mask & n_mask

        orca_loss = self.orca_loss(features, data)
        orca_loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        return orca_loss

    def supervised_loss(self, features, data):
        uncert_margin = self.compute_uncert_margin(features, data)
        uncert_adjusted_features = features + self.uncert_weight * uncert_margin

        out = F.softmax(uncert_adjusted_features, dim=1)
        sup_loss = F.cross_entropy(out[data.labeled_mask], data.y[data.labeled_mask])

        return sup_loss

    def pairwise_loss(self, features, data):
        cosine_dist = F.normalize(features) @ F.normalize(features).t()
        cosine_mat = torch.div(cosine_dist, self.parinwise_temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)
        cosine_mat = cosine_mat - torch.diag(mat_max) #- mat_max.detach()
        sims, indices = torch.min(cosine_mat, dim=1)

        labels = torch.zeros(features.shape[0], features.shape[0], dtype=torch.bool)
        labels[:, indices] = 1
        labels[data.labeled_mask, data.labeled_mask] = torch.eq(data.y[data.labeled_mask], data.y[data.labeled_mask])

        
        F.softmax(cosine_dist, dim = 1)
        


    def compute_uncert_margin(self, features, data):
        uncert_margin = torch.mean(1 - torch.max(F.softmax(-features[data.unlabeled_mask,:], dim=1), dim=1))
        return uncert_margin

    def everything_to_device(self, device):
        return self
        



