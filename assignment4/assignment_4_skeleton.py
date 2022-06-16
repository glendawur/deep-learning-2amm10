#!/usr/bin/env python
# coding: utf-8

# # Group Number:
# 
# # Student 1: Ryan Meghoe
# 
# # Student 2: Nikita Jain
# 
# # Student 3: Andrei Rykov

# Interesting notes about implementation of the VAE with convolutional layers:
# 
# https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

# # Imports

# In[ ]:


import numpy as np
import pickle
import requests

# other imports go here




import torch
import torch.nn as nn
import torch.nn.functional as F

# # Data loading and inspection

# load and inspect data

from torch.utils.data import TensorDataset, DataLoader

data_location = 'https://surfdrive.surf.nl/files/index.php/s/K3ArFDQJb5USQ6K/download'
data_request = requests.get(data_location)
full_data = pickle.loads(data_request.content)

batch_size = 64

unlabeled_dataset = TensorDataset(torch.Tensor(full_data['unlabeled_data']))

classification_dataset = TensorDataset(torch.Tensor(full_data['labeled_data']['data']),
                                       torch.Tensor(full_data['labeled_data']['labels']))

anomaly1_dataset = TensorDataset(torch.Tensor(full_data['representative_set_1']['data']),
                                       torch.Tensor(full_data['representative_set_1']['labels']))
                                       
anomaly2_dataset = TensorDataset(torch.Tensor(full_data['representative_set_2']['data']),
                                       torch.Tensor(full_data['representative_set_2']['labels']))
                                       
                                       
train_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
classification_dataloader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=False)
anomaly1_dataloader = DataLoader(anomaly1_dataset, batch_size=batch_size, shuffle=False)
anomaly2_dataloader = DataLoader(anomaly2_dataset, batch_size=batch_size, shuffle=False)

# Model
class Encoder(nn.Module):
    def __init__(self, z_dim=LATENT_DIM):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # 4x4x256

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn1d = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(4*4*256, 128)
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_var = nn.Linear(128, z_dim)
    
    def forward(self, image):
        x = self.bn1(F.leaky_relu(self.conv1(image)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = self.bn1d(F.leaky_relu(self.fc1(x.view(-1, 4*4*256))))
        mu = self.fc_mu(x)
        var = self.fc_var(x)

        return mu, var

class Decoder(n.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.FC = nn.Linear(latent_dim, 2*2*256)
        self.Conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding =1) # 6x6x128
        self.Conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding =1) # 14x14x64
        self.Conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding =1) # 30x30x32
        self.Conv4 = nn.ConvTranspose2d(32, 1, 3) # 32x32x1


     
    def forward(self,latent_dim):
        x = F.relu(self.FC(latent_dim))
        # x = x.view(-1, 256*4*4)
        x = x.view(-1, 256, 2, 2)
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        # x = F.relu(self.Conv4(x))
        x = torch.sigmoid(self.Conv4(x))
        
        return x

    

class VAE(nn.Module):
    def __init__(self, z_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, input):
        mu, log_var = self.enc(input)
        z = self.reparameterize(mu, log_var)
        output = self.dec(z)
        return output
        


# # Data augmentation and pipeline


# code for data augmentation pipeline 


# # Model definition
# code for model definitions goes here








# # Training and validation loop

def loss_function(x, x_reconstr, mu, log_sigma):
    reconstr_loss = nn.functional.mse_loss(x_reconstr, x, reduction='sum')
    kl_loss = 0.5 * torch.sum(mu.pow(2) + (2*log_sigma).exp() - 2*log_sigma - 1)
    total_loss = reconstr_loss + kl_loss
    return total_loss, reconstr_loss, kl_loss

def train_model(model, dataloader, loss, epochs: int = 50, device = torch.device('cpu')):
    model.train()
    overall_loss = 0
    overall_reconstr_loss = 0
    overall_kl_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            x_reconstr, mu, log_sigma = model(x)
            loss, reconstr_loss, kl_loss = loss(x.view(x.shape[0], -1), x_reconstr.view(x.shape[0], -1), mu, log_sigma)
            
            overall_loss += loss.item()
            overall_reconstr_loss += reconstr_loss.item()
            overall_kl_loss += kl_loss.item()
            
            loss.backward()
            optimizer.step()
            
        n_datapoints = batch_idx * batch_size
    print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / n_datapoints, "\tReconstruction Loss:", overall_reconstr_loss / n_datapoints, "\tKL Loss:", overall_kl_loss / n_datapoints)

from tqdm import tqdm

def elbo_elementwise(x, x_reconstr, mu, log_sigma):
    reconstr_loss = torch.sum(nn.functional.mse_loss(x_reconstr, x, reduction='none'), dim=1)
    constant_term = x.shape[1] * 0.5 * np.log(np.pi)  # assuming sigma=1/sqrt(2)
    kl_loss = 0.5 * torch.sum(mu.pow(2) + (2*log_sigma).exp() - 2*log_sigma - 1, dim=1)
    elbo = - reconstr_loss - constant_term - kl_loss
    return elbo, reconstr_loss, kl_loss

# detection anomalies
def validate_model(model, dataloader, loss, device = torch.device('cpu')):
    model.eval()

    elbo = []
    rec = []
    label = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x_rec, mu, log_sigma = model(x)
            elbo, rec, kl = elbo_elementwise(x.view(x.shape[0], -1), x_rec.view(x.shape[0], -1), mu, log_sigma)
            elbo.append(elbo.cpu().numpy())
            rec.append(rec.cpu().numpy())
            label.append()
    



# perform training


# # Inspection, Validation, and Analysis


# Inspect, validate, and analyse your trained model

