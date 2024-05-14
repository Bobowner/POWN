import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.manifold import TSNE

from open_dataset import load_folds

dataset_name = "photo"
model = "pown"

path = Path("embeddings/embeddings_"+model+"_"+dataset_name+".pth")
embeddings = torch.load(path)
embeddings = embeddings.detach().numpy()

tsne = TSNE(n_components=2, n_jobs=-1)

path = Path("embeddings/tsne_repr_"+model+"_"+dataset_name+".pth")

recuded_repr= tsne.fit_transform(embeddings)
recuded_repr = torch.Tensor(recuded_repr)
torch.save(recuded_repr, path)