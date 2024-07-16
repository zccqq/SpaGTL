# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


class Layer1(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class Layer2(nn.Module):
    
    def __init__(
        self,
        n_in: int = 128,
        n_out: int = 10,
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.var_eps = var_eps
        self.mean_encoder = nn.Linear(n_in, n_out)
        self.var_encoder = nn.Linear(n_in, n_out)
        self.var_activation = torch.exp if var_activation is None else var_activation
        
    def forward(self, x):
        q_m = self.mean_encoder(x)
        q_v = self.var_activation(self.var_encoder(x)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        
        return dist, latent


class Layer3(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class Layer4(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)


class self_attention(nn.Module):
    
    def __init__(
        self,
        n_var: int,
    ):
        super().__init__()
        
        self.n_dim = 10
        self.Q = torch.ones((n_var, self.n_dim))
        self.K = torch.ones((self.n_dim, n_var))
        self.V = torch.nn.Parameter(torch.eye(n_var), requires_grad=False)
        self.QK = torch.nn.Parameter(1.0e-8 * torch.matmul(self.Q, self.K) / self.n_dim)
        
    def forward(self, x):
        torch.diagonal(self.QK.data).fill_(0)
        return torch.matmul(torch.matmul(x, nn.ReLU()(self.QK)), self.V)
        
    def getQK(self):
        torch.diagonal(self.QK.data).fill_(0)
        return nn.ReLU()(self.QK).detach().cpu().numpy()


class SpaGTL(nn.Module):
    
    def __init__(
        self,
        n_input: int,
        n_covar: int,
        n_hidden: int,
        n_latent: int,
        dropout_rate: float = 0.1,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.n_covar = n_covar
        
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        
        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )
        
        self.layer3 = Layer3(
            n_in=n_latent+n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )
        
        self.attention = self_attention(
            n_var=n_input,
        )
    
    def load_pretrained_params(self, params_dict):
        
        self.attention.QK.data = torch.from_numpy(params_dict['QK'])
        self.layer1.network[0].weight.data = torch.from_numpy(params_dict['layer1'])
        self.layer2.mean_encoder.weight.data = torch.from_numpy(params_dict['layer2m'])
        self.layer2.var_encoder.weight.data = torch.from_numpy(params_dict['layer2v'])
        self.layer3.network[0].weight.data = torch.from_numpy(params_dict['layer3'])
        self.layer4.network[0].weight.data = torch.from_numpy(params_dict['layer4'])
    
    def inference(self, x):
        
        x1 = self.layer1(x)
        qz, z = self.layer2(x1)
        
        return dict(x1=x1, z=z, qz=qz)
    
    def generative(self, z, covar):
        
        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))
        
        x4 = self.layer4(x3)
        
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        return dict(x3=x3, x4=x4, pz=pz)
    
    def forward_attention(self, x, covar):
        
        wx = self.attention(x)
        x1 = self.layer1(wx)
        qz, z = self.layer2(x1)
        
        return dict(wx=wx, x1=x1, z=z, qz=qz)
    
    def loss(
        self,
        x,
        inference_outputs,
        generative_outputs,
        W_outputs,
        W_weight
    ):
        
        kl_divergence = kl(inference_outputs['qz'], generative_outputs['pz']).sum(dim=1)
        reconst_loss = torch.norm(x - generative_outputs['x4'])
        loss = (6 - 5 * W_weight) * (torch.mean(kl_divergence) + reconst_loss)
        if W_weight > 0.5:
            loss += W_weight * torch.norm(x - W_outputs['wx'])
            loss += W_weight * torch.norm(inference_outputs['x1'] - W_outputs['x1'])
            loss += W_weight * torch.norm(inference_outputs['qz'].loc - W_outputs['qz'].loc)
            
        return loss



















