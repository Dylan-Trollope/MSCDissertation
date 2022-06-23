import numpy as np 
import torch 
import torch.nn as nn

from torch.autograd import Variable
import torchvision.transforms as T
import time 


class DQN():

	def __init__(self, state_dim, action_dim, lr=0.001):
		self.loss = nn.MSELoss()

		self.nn = nn.Sequential(
				torch.nn.Linear(state_dim, 64),
				nn.ReLU(),
				nn.Linear(64, 128),
				nn.ReLU(),
				nn.Linear(128, action_dim)
		)

		self.optimiser = torch.optim.Adam(self.nn.parameters(), lr)
				
	def update(self, state, y):
		y_pred = self.nn(torch.Tensor(state))
		loss = self.loss(y_pred, Variable(torch.Tensor(y)))
		self.optimiser.zero_grad()
		loss.backward()
		self.optimiser.step()


	def predict(self, state):
		with torch.no_grad():
			return self.nn(torch.Tensor(state))






	
