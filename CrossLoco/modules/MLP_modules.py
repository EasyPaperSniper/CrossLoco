import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class MLP(nn.Module):
    is_recurrent = False
    def __init__(self,  input_dim,
                        ouput_dim,
                        hidden_dims=[256, 256, 256],
                        activation='elu',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MLP, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim = input_dim


        MLP_layers = []
        MLP_layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        MLP_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                MLP_layers.append(nn.Linear(hidden_dims[l], ouput_dim))
            else:
                MLP_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                MLP_layers.append(activation)
        self.MLP = nn.Sequential(*MLP_layers)


        print(f"MLP Structure: {self.MLP}")


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, obs):
        return self.MLP(obs)
    

    

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
