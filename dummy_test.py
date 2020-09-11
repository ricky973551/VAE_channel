import dataset_init
import matplotlib.pyplot as plt
import numpy as np
import vae_utils
import torch
import torch.nn as nn
import seaborn as sns

from sklearn.manifold import TSNE

device = 'cuda'
device = 'cpu'  # uncomment this line to run the model on the CPU
batch_size = 200
# dataset = datasets.MNIST

train_data_set, test_data_set, param_data_set = dataset_init.get_data_set()

if device == 'cuda':
    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=1000, shuffle=True, num_workers=1, pin_memory=True
    )
elif device == 'cpu':
    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=1000, shuffle=True,
    )

obs_dim = 128  # MNIST images are of shape [1, 28, 28]

x, y = next(iter(test_loader))
x = x.view(x.shape[0], obs_dim).to(device)

sns.set_style('whitegrid')
tsne = TSNE(init='pca')
# Dimensionality reduction on the embeddings using t-SNE
emb = tsne.fit_transform(x)

plt.figure(figsize=[10, 7])
labels = y.cpu().numpy()
for i in np.unique(labels):
    class_ind = np.where(labels == i)
    plt.scatter(emb[class_ind, 0], emb[class_ind, 1], label=f'{i}', alpha=0.5)
    plt.legend()
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
