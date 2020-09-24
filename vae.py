import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import dataset_init
from sklearn.decomposition import PCA


# These packages are required by the visualization utils
import seaborn as sns
from sklearn.manifold import TSNE

import vae_utils


class VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        """Initialize the VAE model.

        Args:
            obs_dim: Dimension of the observed data x, int
            latent_dim: Dimension of the latent variable z, int
            hidden_dim: Hidden dimension of the encoder/decoder networks, int
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        # Trainable layers of the encoder
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear21 = nn.Linear(hidden_dim, latent_dim)
        self.linear22 = nn.Linear(hidden_dim, latent_dim)
        # Trainable layers of the decoder
        self.linear3 = nn.Linear(latent_dim, hidden_dim)
        self.linear41 = nn.Linear(hidden_dim, obs_dim)
        self.linear42 = nn.Linear(hidden_dim, obs_dim)

    def encoder(self, x):
        """Obtain the parameters of q(z) for a batch of data points.

        Args:
            x: Batch of data points, shape [batch_size(also: b), obs_dim]

        Returns:
            mu: Means of q(z), shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q(z), shape [batch_size, latent_dim]
        """
        ##########################################################
        # YOUR CODE HERE

        h_relu = torch.relu(self.linear1(x))
        mu = self.linear21(h_relu)
        logsigma = -torch.relu(self.linear22(h_relu))

        return mu, logsigma

        ##########################################################

    def sample_with_reparam(self, mu, logsigma):
        """Draw sample from q(z) with reparametrization.

        We draw a single sample z_i for each data point x_i.

        Args:
            mu: Means of q(z) for the batch, shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q(z) for the batch, shape [batch_size, latent_dim]

        Returns:
            z: Latent variables samples from q(z), shape [batch_size, latent_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        batch_size, latent_dim = mu.shape
        eps = torch.normal(0, 1, size=(batch_size, latent_dim))

        sigma = torch.exp(logsigma)
        z = sigma * eps + mu

        return z

        ##########################################################

    def decoder(self, z):
        """Convert sampled latent variables z into observations x.

        Args:
            z: Sampled latent variables, shape [batch_size/num_samples, latent_dim]

        Returns:
            theta: Parameters of the conditional likelihood, shape [batch_size/num_samples, obs_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        h_relu = torch.relu(self.linear3(z))
        theta = self.linear41(h_relu)
        logsigma_d = -torch.relu(self.linear42(h_relu))

        return theta, logsigma_d

        ##########################################################

    def kl_divergence(self, mu, logsigma):
        """Compute KL divergence KL(q_i(z)||p(z)) for each q_i in the batch.

        Args:
            mu: Means of the q_i distributions, shape [batch_size, latent_dim]
            logsigma: Logarithm of standard deviations of the q_i distributions,
                      shape [batch_size, latent_dim]

        Returns:
            kl: KL divergence for each of the q_i distributions, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        sigma = torch.exp(logsigma)
        pre_kl = sigma ** 2 + mu ** 2 - 2 * logsigma - 1

        kl = 0.5 * torch.sum(pre_kl, dim=1)

        return kl

        ##########################################################

    def elbo(self, x):
        """Estimate the ELBO for the mini-batch of data.

        Args:
            x: Mini-batch of the observations, shape [batch_size, obs_dim]

        Returns:
            elbo_mc: MC estimate of ELBO for each sample in the mini-batch, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE

        b, obs_dim = x.shape
        mu, logsigma = self.encoder(x)
        z = self.sample_with_reparam(mu, logsigma)
        theta, logsigma_d = self.decoder(z)
        kl = self.kl_divergence(mu, logsigma)

        # log_px_ifz = torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta), dim=1)
        log_px_ifz = torch.sum(
            (-0.5 * np.log(2.0 * np.pi))
            + (-1 * logsigma_d)
            + ((-0.5 / (torch.exp(logsigma_d)**2)) * (x - theta) ** 2.0),
            dim=1,
        )

        return log_px_ifz, kl, mu, logsigma, theta

        ##########################################################

    def sample(self, num_samples, mu, logsigma):
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate.
        Returns:
            samples: Samples generated by the model
        """
        ##########################################################
        # YOUR CODE HERE
        zp = torch.normal(0, 2, size=(num_samples, latent_dim))
        theta, logsigma_d = self.decoder(zp)
        # print(logsigma_d)
        # print(theta)
        samples = self.sample_with_reparam(theta, logsigma_d)

        return samples

        ##########################################################

    def elbo_with_filter(self, x):
        """Only to be used in eval mode!
        Used to calculate the ELBO of generated channels.
        If ELBO is to low i.e. the ELBO is not maximized for that specific channel
        it will be discarded (filtered out).

        Args:
            x: Mini-batch of the observations, shape [batch_size, obs_dim]

        Returns:
            elbo_mc: MC estimate of ELBO for each sample in the mini-batch, shape [batch_size]

        """
        ##########################################################





device = 'cuda'
device = 'cpu'  # uncomment this line to run the model on the CPU
batch_size = 100

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

print('debug')

obs_dim = 128
latent_dim = 16  # Size of the latent variable z
hidden_dim = 64  # Size of the hidden layer in the encoder / decoder

# dataset = datasets.MNIST
# if device == 'cuda':
#     train_loader = torch.utils.data.DataLoader(
#         dataset('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
#                                                                                  transforms.Normalize(0, 1)])),
#         batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
#                                                                                  transforms.Normalize(0, 1)])),
#         batch_size=1000, shuffle=True, num_workers=1, pin_memory=True
#     )
# elif device == 'cpu':
#     train_loader = torch.utils.data.DataLoader(
#         dataset('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
#                                                                                  transforms.Normalize(0, 1)])),
#         batch_size=batch_size, shuffle=True,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
#                                                                                  transforms.Normalize(0, 1)])),
#         batch_size=1000, shuffle=True,
#     )
#
# obs_dim = 784  # MNIST images are of shape [1, 28, 28]
# latent_dim = 32  # Size of the latent variable z
# hidden_dim = 400  # Size of the hidden layer in the encoder / decoder

vae = VAE(obs_dim, latent_dim, hidden_dim).to(device)
opt = torch.optim.Adam(vae.parameters(), lr=1e-5)

max_epochs = 100
display_step = 100
annealing = 0.0
constant = 5.0
n_gen_samples = 500
# init list for losses
loss_hist = []
recon_loss_hist = []
kld_hist = []

for epoch in range(max_epochs):
    print(f'Epoch {epoch}')
    print(f'KLD weight {annealing}')
    for ix, batch in enumerate(train_loader):
        x, y = batch
        opt.zero_grad()
        # We want to maximize the ELBO, so we minimize the negative ELBO

        recon_loss, kld, mu, logsigma, theta = vae.elbo(x)
        loss = recon_loss - annealing * (kld - constant)  # actual calculation of ELBO aka loss
        loss = -loss.mean(-1)
        loss.backward()
        opt.step()

    if annealing < 1.0:
        annealing = annealing + 0.05

    recon_loss = recon_loss.mean(-1)
    kld = kld.mean(-1)

    loss_hist.append(loss.item())
    recon_loss_hist.append(-recon_loss.item())
    kld_hist.append(kld.item())

    print(f'  loss i.e. neg ELBO = {loss.item():.4f}')
    print(f'  neg reconstruction error = {-recon_loss.item():.4f}')
    print(f'  kl divergence = {kld.item():.4f}')

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Loss', fontsize=16)
ax = fig.add_subplot(111)
plt.plot(range(max_epochs), loss_hist, 'b', label='Negative ELBO Loss')
plt.plot(range(max_epochs), recon_loss_hist, 'g', label='Reconstruction Loss')
plt.plot(range(max_epochs), kld_hist, 'r', label='KL Divergence')
plt.legend()

# load new channels from TEST data set
x, y = next(iter(test_loader))

# Visualize Latent Space Embeddings
vae_utils.visualize_embeddings(vae, x, y)

vae.eval()

with torch.no_grad():
    # Generation of Channels
    x_sample = vae.sample(n_gen_samples, mu, logsigma).detach().cpu()
    y_sample = 8 * np.ones((n_gen_samples, 1), dtype=int)

    # vae_utils.visualize_vae_samples(x_sample)

    # Reconstruction of Input
    x_recon = vae.elbo(x)[4]
    y_recon = y + 4

    # Reconstruction + Filtering of Generated samples
    condition = loss.item() * 0.9
    srecon_loss = vae.elbo(x_sample)[0]  # s for sample
    skl = vae.elbo(x_sample)[1]
    x_rsample = vae.elbo(x_sample)[4]
    elbo_s = srecon_loss - skl
    idx = torch.where(-elbo_s > condition)
    print(idx[0])

    # only select indices that fulfil condition
    x_fsample = x_rsample[idx[0], :]
    n_fsamples = x_fsample.shape[0]
    y_fsample = 8 * np.ones((n_fsamples, 1), dtype=int)

    # Plotting everything together in same TSNE environment
    vae_utils.visualize_all(vae, x, y, x_recon, y_recon, x_fsample.numpy(), y_fsample)


plt.show()
