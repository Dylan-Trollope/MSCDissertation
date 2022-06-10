import torch
import torch.nn as nn
import torch.autograd as autograd


class DQN(nn.Module):


    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dium = input_dim
        self.output_dim = output_dim

        self.cn = nn.Sequential(
                nn.Linear(self.input_dim[0], 128),
                nn.ReLu(),
                nn.Linear(128, 256),
                nn.ReLu(),
                nn.Linear(256, self.output_dim)
        )


    def forward(self, state):
        q_vals = self.cn(state)
        return q_vals



 # Need to do COnvDQN over here

