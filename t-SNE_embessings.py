import torch
from pathlib import Path
from sklearn.manifold import TSNE

path = Path("embeddings/embeddings_gcn_ogb-arxiv.pth")
embeddings = torch.load(path)
embeddings = embeddings.detach().numpy()

tsne = TSNE(n_components=2, n_jobs=-1)
recuded_repr= tsne.fit_transform(embeddings)

recuded_repr = torch.Tensor(recuded_repr)
path = Path("embeddings/tsne_repr_gcn_ogb-arxiv.pth")
