import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from typing import Tuple
import math


gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")
print("device:", device)


# encoder layer. Takes x as input and encodes it to a latent space
class LadderEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z_dim = latent_dim

        # hidden layers
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.batchnorm = nn.BatchNorm1d(self.out_dim)

        # latent layers
        self.mu = nn.Linear(self.out_dim, self.z_dim)
        self.var = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x):
        # hidden
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)

        # latent
        mu = self.mu(x)
        var = F.softplus(self.var(x))

        return x, mu, var


# decoder layer.
class LadderDecoder(nn.Module):
    def __init__(self, z1_dim, hidden_dim, z2_dim):
        super().__init__()

        self.linear = nn.Linear(z1_dim, hidden_dim)
        self.batchnrom = nn.BatchNorm1d(hidden_dim)

        self.mu = nn.Linear(hidden_dim, z2_dim)
        self.var = nn.Linear(hidden_dim, z2_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnrom(x), 0.1)

        mu = self.mu(x)
        var = F.softplus(self.var(x))

        return mu, var


# Final decoder layer.
class FinalLadderDecoder(nn.Module):
    def __init__(self, z_final, hidden_dim, input_dim):
        super().__init__()

        self.linear = nn.Linear(z_final, hidden_dim)
        self.recon = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return F.sigmoid(self.recon(x))
    


class LadderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dims):
        super().__init__()
        self.device = device
        self.data_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims

        self.n_layers = len(latent_dims)

        neurons = [input_dim, *hidden_dims]

        encoder_layers = [LadderEncoder(neurons[i-1], neurons[i], latent_dims[i-1]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder(latent_dims[i], hidden_dims[i], latent_dims[i-1]) for i in range(1, len(hidden_dims))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.recon = FinalLadderDecoder(latent_dims[0], hidden_dims[0], input_dim)

    # function to reparameterize. Same as in the standard VAE
    def reparameterize(self, mu, var):
        eps = torch.randn_like(var)
        std = torch.sqrt(var)
        return mu + eps * std

    # function to merge two gaussians, taken from the paper
    def merge_gaussian(self, mu1, var1, mu2, var2):
        precision1 = 1 / (var1 + 1e-8)
        precision2 = 1 / (var2 + 1e-8)

        new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)
        new_var = 1 / (precision1 + precision2)
        return new_mu, new_var

    # function to compute the KL divergence
    def compute_kl(self, z, q_params, p_params = None):
        (mu, var) = q_params

        # compute log of gaussian
        qz = torch.sum(- 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var + 1e-8) - (z - mu)**2 / (2 * var + 1e-8), dim=-1)

        if p_params is None:
            # log of standard gaussian
            pz = torch.sum(-0.5 * math.log(2 * math.pi) - z**2 /2, dim=-1)
        else:
            (mu, var) = p_params
            pz = torch.sum(- 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var + 1e-8) - (z - mu)**2 / (2 * var + 1e-8), dim=-1)

        return qz - pz

    def forward(self, x):

        mu_var_lst = []
        latents = []

        # deterministic upward pass over all the encoding layers
        for layer in self.encoder:
            x, mu, var = layer(x)
            mu_var_lst.append((mu, var))

        # reparameterize using mu and var from last latent layer
        mu, var = mu_var_lst[-1]
        z = self.reparameterize(mu, var)
        latents.append(z)
        mu_var_lst = list(reversed(mu_var_lst))

        self.kl_divergence = 0
        self.kl_divergence_per_layer = []

        # stochastic downward pass recursively computing
        # both the approximate posterior and generative distributions
        for i, decoder in enumerate([-1, *self.decoder]):

            mu_d, var_d = mu_var_lst[i]

            if i == 0:
                # we are at the top, we have to compute the kl
                layer_kl = self.compute_kl(z, (mu_d, var_d))
                self.kl_divergence += layer_kl
                self.kl_divergence_per_layer.append(torch.sum(layer_kl))

            else:
                # otherwise we have to pass the z through the decoder
                # get the mu and var and merge them with the one we get in the previous step
                mu_t, var_t = decoder(z)
                merged_mu, merged_var = self.merge_gaussian(mu_d, var_d, mu_t, var_t)

                # and now we can sample them
                z = self.reparameterize(merged_mu, merged_var)
                latents.append(z)

                # and compute the kl
                layer_kl = self.compute_kl(z, (merged_mu, merged_var), (mu_t, var_t))
                self.kl_divergence += layer_kl
                self.kl_divergence_per_layer.append(torch.sum(layer_kl))

        # final decoder, i.e. reconstruction
        recon = self.recon(z)
        return recon, latents

    # function to sample
    def decode(self, z):
        for decoder in self.decoder:
            mu, var = decoder(z)
            z = self.reparameterize(mu, var)

        return self.recon(z)



class ConvLayer(nn.Conv2d):
    """
    Serves as a convolutional layer in the Diffusion NN

    Args:
        x: (batch_size, C_in, H, W)
    Returns:
        y: (batch_size, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        
        padding = kernel_size // 2 * dilation
        
        super().__init__(in_channels, out_channels, kernel_size, padding=padding,
                         stride=stride, dilation=dilation, groups=groups, bias=bias)
        
        self.group_norm = nn.GroupNorm(8, out_channels)
        self.activation_fn = nn.SiLU()
        
    def forward(self, x, t):
            x = x + t
            temp = x
            x = super(ConvLayer, self).forward(x)
            x = temp + x
            x = self.group_norm(x)
            x = self.activation_fn(x)
            return x
    

class DiffusionNet(nn.Module):
    def __init__(self, T_steps=1000, image_size=[1, 28, 28], hidden_dims=[256, 256], temb_dim=256):
        super().__init__()
        self.device = device
        self.T=T_steps
        self.img_C, self.img_H, self.img_W = image_size
        self.hidden_dims = hidden_dims
        self.temb_dim = temb_dim

        self.betas = torch.linspace(0.0001, 0.02, T_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = self.alphas.cumprod(0)

        
        # input layer
        self.in_conv1 = nn.Conv2d(in_channels=self.img_C, out_channels=hidden_dims[0], kernel_size=7, padding=3)
        
        # time embedding layer
        self.t_layer1 = nn.Conv2d(in_channels=temb_dim, out_channels=hidden_dims[0], kernel_size=1, padding=0)
        self.t_layer2 = nn.SiLU()
        self.t_layer3 = nn.Conv2d(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=1, padding=0)
        
        # middle layers
        self.mid_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_layers = nn.ModuleList([])
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(ConvLayer(hidden_dims[i-1], hidden_dims[i], kernel_size=3, dilation=3**((i-1)//2)))
        
        # output layer
        self.out_conv1 = nn.Conv2d(in_channels=256, out_channels=self.img_C, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        
        # embedding
        t_embedding = self.pos_encoding(t, self.temb_dim)
        t_embedding = self.t_layer2(self.t_layer1(t_embedding.unsqueeze(-1).unsqueeze(-2)))
        t_embedding = self.t_layer3(t_embedding)

        # input layer
        x = self.in_conv1(x)
        
        # second layer (first mid layer, no activation func)
        x = x + t_embedding
        temp = x
        x = self.mid_conv1(x)
        x = temp + x
        
        # middle layers
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, t_embedding)
        
        # output layer
        x = self.out_conv1(x)

        return x
        
    # scale to -1 to 1, crucial according to DDPM paper.
    def minus_one_one(self, x):
        return x * 2 - 1
    
    def zero_one(self, x):
        return (x + 1) * 0.5

    # function for time embedding, used in the forward part of the NN
    def pos_encoding(self, t, channels):
        device = t.device
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    # add noise
    def add_noise(self, X, t):
        X = self.minus_one_one(X)
        epsilon = torch.normal(0., 1., size=X.shape).to(self.device)
        
        alpha_bar = self.alpha_bars[t][:, None, None, None]
        noisy_sample = X * torch.sqrt(alpha_bar) + epsilon * torch.sqrt(1 - alpha_bar)
        return noisy_sample
        
    
    # loss function
    def loss(self, X):
        X = self.minus_one_one(X)
        ts = torch.randint(self.T, size=(X.shape[0],)).to(device) # pick t uniformly at random
        alpha = self.alpha_bars[ts][:, None, None, None]
        epsilon = torch.normal(0., 1., size=X.shape).to(device)
        samples = torch.sqrt(alpha)*X + torch.sqrt(1-alpha)*epsilon
        eps_model = self.forward(samples, ts)
        return nn.functional.mse_loss(eps_model, epsilon)
        
    # function to generate samples
    def sample(self, num, img=None):
        if img is None:
            X = torch.normal(0.0, 1.0, size=(num, self.img_C, self.img_H, self.img_W), dtype=torch.float32).to(device)
        else:
            X = img

        for t in reversed(range(self.T)):
            timestep = torch.tensor([t]).repeat(num).to(self.device)

            if t > 1:
              z = torch.randn_like(X).to(self.device)
            else:
              z = torch.zeros_like(X).to(self.device)
            beta = self.betas[timestep][:, None, None, None]
            alpha = self.alphas[timestep][:, None, None, None]
            alpha_bar = self.alpha_bars[timestep][:, None, None, None]
            eps_model = self.forward(X, timestep)
            X = 1 / torch.sqrt(alpha) * (X - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_model) + z * torch.sqrt(beta)
        X = self.zero_one(X.clamp(-1, 1))
        return X

