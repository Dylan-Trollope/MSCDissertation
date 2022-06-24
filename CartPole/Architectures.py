import numpy as np 
import torch 
import torch.nn as nn

from torch.autograd import Variable
import torchvision.transforms as T
import random
import time 


class DQN():

	def __init__(self, state_dim, action_dim, lr=0.001):
		self.loss = nn.MSELoss()

		self.nn = nn.Sequential(
				torch.nn.Linear(state_dim, 128),
				nn.LeakyReLU(),
				nn.Linear(128, 256),
				nn.LeakyReLU(),
				nn.Linear(256, action_dim)
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



class ER_DQN(DQN):

	def replay(self, memory, size, gamma):

		if len(memory)>=size:
			batch = random.sample(memory, size)
			batch_t = list(map(list, zip(*batch))) #Transpose batch list
			states = batch_t[0]
			actions = batch_t[1]
			next_states = batch_t[2]
			rewards = batch_t[3]
			is_dones = batch_t[4]

			states = torch.Tensor(states)
			actions_tensor = torch.Tensor(actions).int()
			next_states = torch.Tensor(next_states)
			rewards = torch.Tensor(rewards)
			is_dones_tensor = torch.Tensor(is_dones)

			is_dones_indices = torch.where(is_dones_tensor==True)[0]

			all_q_values = self.nn(states) # predicted q_values of all states
			all_q_values_next = self.nn(next_states)
			
			all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
			all_q_values[is_dones_indices, actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
			self.update(states, all_q_values)




