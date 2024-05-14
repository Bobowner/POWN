import torch
import wandb
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans

class OpenSSL(torch.nn.Module):
    
    def __init__(self, encoder, ssl, out_dim, log_all, device):
        
        super().__init__()
        
        self.encoder = encoder
        self.ssl = ssl
        self.out_dim = out_dim

        self.log_all = log_all
        self.device = device


    def ssl_loss(self, x, edge_index):
        
        pos_z, neg_z, summary = self.ssl.forward(x, edge_index)
        
        loss = self.ssl.loss(pos_z, neg_z, summary)

        return loss

    def forward(self, x, edge_index):
        
        logits = self.encoder(x, edge_index)
        embeddings = logits.detach().cpu().numpy()
                
        kmeans = KMeans(n_clusters=self.out_dim, init="k-means++", n_init=1).fit(embeddings)

        prototypes = torch.Tensor(kmeans.cluster_centers_).to(self.device)
        distances = torch.cdist(logits,prototypes)
        probas = F.softmax(-distances, dim=1)
        
        return probas


    def inference(self, x, edge_index):
        probas = self.forward(x, edge_index)
        return probas.argmax(dim=1)


    def train_one_epoch(self, optimizer, data):
    
        #mapped_labels = torch.Tensor([input_class_mapping[label.item()] for label in data.y])
        
        optimizer.zero_grad()
        
        loss = self.ssl_loss(data.x, data.edge_index)
        loss.backward()
    
        if self.log_all:        
            wandb.log({'ssl_loss': loss})
    
        optimizer.step()
    
        return loss

    def everything_to_device(self, device):
        return self

    def post_train_processing(self, data):
        return self