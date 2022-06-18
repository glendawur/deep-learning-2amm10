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

from torch.utils.data import TensorDataset, DataLoader, Dataset

data_location = 'https://surfdrive.surf.nl/files/index.php/s/K3ArFDQJb5USQ6K/download'
data_request = requests.get(data_location)
full_data = pickle.loads(data_request.content)

batch_size = 64



class FashionMNISTDataset(Dataset):
    def __init__(self, data, targets = None, transform=None):
        self.data = torch.Tensor(data)

        if targets is None:
            self.targets = torch.Tensor(torch.zeros(data.shape[0], 1))
        else:
            assert data.shape[0] == targets.shape[0]
            self.targets = torch.Tensor(targets)

        self.transform = transform
        
    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.data[index])
        else:
            x = self.data[index]
        return x, self.targets[index]

    def __len__(self):
        return len(self.data)

    def __shape__(self):
        return self.data.shape, self.targets.shape

from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

def train_validation_split(X: torch.Tensor, Y: torch.Tensor, train_size = 0.8, transform = None):
    """
    The function that returns the train and validation datasets
    Args:
        X (np.ndarray): the feature matrix
        Y (np.ndarray): the labels matrix
    
    Output:
        Train (TrainProcessDataset): the object of class TrainProcessDataset with train data
        Validation (TrainProcessDataset): the object of class TrainProcessDataset with validation data
    """
    indices = list(SubsetRandomSampler(range(X.shape[0])))
    train_X, train_Y = X[:int(np.floor(len(indices)*train_size))], Y[:int(np.floor(len(indices)*train_size))]
    val_X, val_Y = X[int(np.floor(len(indices)*train_size)):], Y[int(np.floor(len(indices)*train_size)):]
    return FashionMNISTDataset(train_X, train_Y, transform), FashionMNISTDataset(val_X, val_Y)


augmentation = transforms.RandomAffine(degrees = (-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.1), fill  = 0)

unlabeled_dataset = FashionMNISTDataset(torch.Tensor(full_data['unlabeled_data']), transform=augmentation)

classification_train, classification_test = train_validation_split(torch.Tensor(full_data['labeled_data']['data']),
                                                                   torch.Tensor(full_data['labeled_data']['labels']),
                                                                   transform=augmentation)

anomaly1_dataset = TensorDataset(torch.Tensor(full_data['representative_set_1']['data']),
                                 torch.Tensor(full_data['representative_set_1']['labels']))
                                       
anomaly2_dataset = TensorDataset(torch.Tensor(full_data['representative_set_2']['data']),
                                 torch.Tensor(full_data['representative_set_2']['labels']))
                                       

# the dataloader with 28000 instances to train the VAE                                       
train_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

# the dataloader for the classification task
classification_test_dataloader = DataLoader(classification_train, batch_size=batch_size, shuffle=False)
classification_train_dataloader = DataLoader(classification_test, batch_size=batch_size, shuffle=False)

# the dataloaders for the anomaly detection tasks
anomaly1_dataloader = DataLoader(anomaly1_dataset, batch_size=batch_size, shuffle=False)
anomaly2_dataloader = DataLoader(anomaly2_dataset, batch_size=batch_size, shuffle=False)

LATENT_DIM = 16

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

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
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


class Classificator(nn.Module):
    def __init__(self, encoder, latent_dim=LATENT_DIM, n_classes=5):
        super(Classificator, self).__init__()

        self.encoder = encoder

        self.fc = nn.Linear(latent_dim, n_classes)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
     
    def forward(self, image):
        mu, logvar = self.encoder(image)
        z = self.reparameterize(mu, logvar)
        out = torch.sigmoid(self.fc(z))
        
        return out, z

    

class VAE(nn.Module):
    def __init__(self, device, z_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.enc = Encoder(z_dim).to(device)
        self.dec = Decoder(z_dim).to(device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, input):
        mu, log_var = self.enc(input)
        z = self.reparameterize(mu, log_var)
        output = self.dec(z)
        return output, mu, log_var
        


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

def train_classifier(encoder, dataloader, criterion, latent_dim=16, epochs: int = 50, device = torch.device('cpu')):
    encoder = encoder.to(device)
    encoder.eval()

    model = Classificator(encoder, latent_dim).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epochs):
        correct = 0
        loss_value = 0
        length = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)


            optimizer.zero_grad()

            y_pred, noise = model(x)
            loss = criterion(y_pred, y)
            loss_value += loss.item()

            correct += (torch.round(y_pred) == y).all(dim=1).sum().item()
            length += batch_size
            
            loss.backward()
            optimizer.step()
            
        n_datapoints = batch_idx * batch_size
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", loss_value / n_datapoints, '\tCorrect labeled percentage: ', correct / length)
    return model

from tqdm import tqdm

def train_model(model, dataloader, epochs: int = 50, device = torch.device('cpu')):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epochs):

        overall_loss = 0
        overall_reconstr_loss = 0
        overall_kl_loss = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            x_reconstr, mu, log_sigma = model(x)
            loss, reconstr_loss, kl_loss = loss_function(x.view(x.shape[0], -1), x_reconstr.view(x.shape[0], -1), mu, log_sigma)
            
            overall_loss += loss.item()
            overall_reconstr_loss += reconstr_loss.item()
            overall_kl_loss += kl_loss.item()
            
            loss.backward()
            optimizer.step()
            
        n_datapoints = batch_idx * batch_size
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / n_datapoints, "\tReconstruction Loss:", overall_reconstr_loss / n_datapoints, "\tKL Loss:", overall_kl_loss / n_datapoints)

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def elbo_elementwise(x, x_reconstr, mu, log_sigma):
    reconstr_loss = torch.sum(nn.functional.mse_loss(x_reconstr, x, reduction='none'), dim=1)
    constant_term = x.shape[1] * 0.5 * np.log(np.pi)  # assuming sigma=1/sqrt(2)
    kl_loss = 0.5 * torch.sum(mu.pow(2) + (2*log_sigma).exp() - 2*log_sigma - 1, dim=1)
    elbo = - reconstr_loss - constant_term - kl_loss
    return elbo, reconstr_loss, kl_loss
    
def anomaly_detection(model, train_dataloader, target_dataloader, device = torch.device('cpu')):
    model.eval()

    elbo_train = []
    elbo_target = []
    rec_train = []
    rec_target = []
    label_target = []

    with torch.no_grad():
        for x, _ in train_dataloader:
            x = x.to(device)
            x_rec, mu, log_sigma = model(x)
            elbo, rec, kl = elbo_elementwise(x.view(x.shape[0], -1), x_rec.view(x.shape[0], -1), mu, log_sigma)
            elbo_train.append(elbo.cpu().numpy())
            rec_train.append(rec.cpu().numpy())
        
        for x, y in target_dataloader:
            x = x.to(device)
            x_rec, mu, log_sigma = model(x)
            elbo, rec, kl = elbo_elementwise(x.view(x.shape[0], -1), x_rec.view(x.shape[0], -1), mu, log_sigma)
            elbo_target.append(elbo.cpu().numpy())
            rec_target.append(rec.cpu().numpy())
            # anomaly class is one in the label list
            label_target.append((y.cpu().numpy().argmax(axis = 1) > 4))

    elbo_train = np.concatenate(elbo_train, 0)
    elbo_target = np.concatenate(elbo_target, 0)
    rec_train = np.concatenate(rec_train, 0)
    rec_target = np.concatenate(rec_target, 0)
    label_target = np.concatenate(label_target, 0)

    fig, axs = plt.subplots(1, 2)
    axs[0,0].hist(elbo_train, bins=100, color = '#99ccff')
    axs[0,0].hist(elbo_target[np.where(label_target == False)], bins=100, color = '#bbeeff')
    axs[0,0].hist(elbo_target[np.where(label_target == True)], bins=100, color = '#cb2c31')
    axs[0,0].set_title('ELBO')
    
    axs[0,1].hist(rec_train, bins=100, color = '#99ccff')
    axs[0,1].hist(rec_target[np.where(label_target == False)], bins=100, color = '#bbeeff')
    axs[0,1].hist(rec_target[np.where(label_target == True)], bins=100, color = '#cb2c31')
    axs[0,1].set_title('Reconstruction Loss')
    fig.show()

    scores = -elbo_target
    scores = (scores - scores.min())/(scores.max()-scores.min())

    fpr, tpr, roc_thresholds = roc_curve(label_target, scores)
    precision, recall, pr_thresholds = precision_recall_curve(label_target, scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    print(f'{roc_auc=}')
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for anomaly detection')
    plt.legend(loc='lower right')
    plt.show()


    print(f'{pr_auc=}')
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for anomaly detection')
    plt.legend(loc='lower left')
    plt.show() 


# perform training


# # Inspection, Validation, and Analysis


# Inspect, validate, and analyse your trained model

