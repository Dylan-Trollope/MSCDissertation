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
        vals = self.cn(state)
        return vals


class ConvolutionalDQN(nn.Module, in_dims, out_dims):
	
	def __init__(self, in_dims, out_dims):
		super(ConvolutionalDQN, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.features = self.feature_dims()

		self.conv_net = nn.Sequential(
				nn.Conv2d(self.in_dims, 32, kernel_size=8, stride=4),
				nn.ReLu(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLu(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLu()
		)

		self.feature_net = nn.Sequential(
				self.Linear(self.features, 128),
				self.ReLU(),
				self.Linear(128, 256),
				nn.ReLU(),
				nn.Linear(256, self.out_dims)
		)



	def forward(self, state):
		features = self.conv_net(state)
		features = features.view(features.size(0), -1)
		q_vals = self.feature_net(features)
		return q_vals
		

	
	def feature_dims():
		return self.conv_net(autograd.Variable(torch.zeros(1, *self.in_dims))).view(1, -1).size(1)
		


