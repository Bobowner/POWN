import torch
import wandb
import torch.nn.functional as F

from torch_geometric.utils import to_networkx

from sklearn.cluster import SpectralClustering
from networkx import adjacency_matrix
from scipy.sparse import csr_matrix






class OpenSpectral(torch.nn.Module):
    def __init__(self, out_dim, known_classes, unknown_classes, 
                 device, log_all=True):


        super().__init__()

        self.n_clusters = out_dim
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes

        self.sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed_nearest_neighbors', assign_labels ="kmeans", n_jobs=-1)
        
        self.log_all = log_all
        self.device = device

    def forward(self, x, edge_index):
        return torch.Tensor(self.sc.labels_).to(self.device)


    def inference(self, x, edge_index):
        return torch.Tensor(self.sc.labels_).to(self.device)
        

    def train_one_epoch(self, optimizer, data):

        #mapped_labels = torch.Tensor([input_class_mapping[label.item()] for label in data.y])
        data = data.cpu()
        nx_graph = to_networkx(data)

        adj_mat = adjacency_matrix(nx_graph)
        adj_mat = csr_matrix(adj_mat)
        adj_mat = adj_mat.toarray()
        
        self.sc.fit(adj_mat)

        return 0

    def everything_to_device(self, device):
        return self

    def post_train_processing(self, data):
        return self
        