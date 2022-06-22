import torch
import torch.nn as nn
import torch.autograd as autograd


class DQN(nn.Module):


    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cn = nn.Sequential(
                nn.Linear(self.input_dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        vals = self.cn(state)
        return vals


class ConvolutionalDQN(nn.Module):
	
	def __init__(self, in_dims, out_dims):
		super(ConvolutionalDQN, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims

		self.conv = nn.Sequential(
				nn.Conv2d(self.in_dims, 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU()
		)

		self.features = self.feature_dims()


		self.feature_net = nn.Sequential(
				self.Linear(self.features, 128),
				self.ReLU(),
				self.Linear(128, 256),
				nn.ReLU(),
				nn.Linear(256, self.out_dims)
		)



	def forward(self, state):
		features = self.conv(state)
		features = features.view(features.size(0), -1)
		q_vals = self.feature_net(features)
		return q_vals
		

	def feature_dims(self):
		return self.conv(autograd.Variable(torch.zeros(1, self.in_dims))).view(1, -1).size(0)
		


