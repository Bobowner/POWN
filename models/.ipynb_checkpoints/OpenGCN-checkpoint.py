import torch
import wandb
import torch.nn.functional as F
from torch_geometric.nn.models import GCN




class OpenGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_dim, dropout, num_layers, known_classes, unknown_classes, 
                 device, log_all=True):


        super().__init__()
        
        self.encoder = GCN(in_channels = in_channels,
              hidden_channels = hidden_channels,
              out_channels = out_dim,
              dropout = dropout,
              num_layers = num_layers)

        self.log_all = log_all
        self.device = device

        #Mapping 0 to n to knonwn classes
        output_class_mapping = {i: known_classes[i] for i in range(len(known_classes))}

        #Mapping known classes to 0 to n
        input_class_mapping = {v: k for k, v in output_class_mapping.items()}


    def forward(self, x, edge_index):
        logits = self.encoder(x, edge_index)
        return F.softmax(logits, dim=1)

    def inference(self, x, edge_index):
        probas = self.forward(x, edge_index)
        return probas.argmax(dim=1)


    def get_embeddings(self, x, edge_index):
        features = self.encoder(x, edge_index)
        return features
        

    def train_one_epoch(self, optimizer, data):

        #mapped_labels = torch.Tensor([input_class_mapping[label.item()] for label in data.y])
        
        optimizer.zero_grad()

        out = self.forward(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.labeled_mask], data.y[data.labeled_mask])
        loss.backward()

        if self.log_all:        
            wandb.log({'cross_entropy_loss': loss})

        optimizer.step()

        return loss

    def everything_to_device(self, device):
        return self

    def post_train_processing(self, data):
        return self
        