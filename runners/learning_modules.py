import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class Xmorph_network(nn.Module):
    def __init__(
        self,
        src_dim,
        tgt_dim,
        mapper_dims= [512, 128, 512],
        activation = 'lrelu',
        **kwargs
    ):
        super().__init__()
        activation = get_activation(activation)
        self.src2tgt_mapper = nn.Sequential(*build_mlp(src_dim, tgt_dim, mapper_dims, activation))
        self.tgt2src_mapper = nn.Sequential(*build_mlp(tgt_dim, src_dim, mapper_dims, activation))

    def forward(self, input):
        return self.src2tgt_mapper(input)
    
    def inverse(self, input):
        return self.tgt2src_mapper(input)
    
    def cycle_forward(self,input):
        mid_output = self.forward(input)
        output = self.inverse(mid_output)
        return mid_output, output



def build_mlp(input_dim, output_dim, hidden_dims, activation):
    actor_layers = []
    actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
    actor_layers.append(activation)
    for layer_index in range(len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            actor_layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
        else:
            actor_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            actor_layers.append(activation)
    return actor_layers

    

    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
