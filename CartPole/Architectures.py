import torch 
import torch.nn as nn

from torch.autograd import Variable
import random
import copy


class DQN(nn.Module):

	def __init__(self, state_dim, action_dim, lr):
		super(DQN, self).__init__()
		self.loss = nn.MSELoss()
		SIZE = 64
		self.nn = nn.Sequential(
				torch.nn.Linear(state_dim, SIZE),
				nn.LeakyReLU(),
				nn.Linear(SIZE, SIZE * 2),
				nn.LeakyReLU(),
				nn.Linear(SIZE * 2, action_dim)
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
			



class ErDQN(DQN):

	def replay(self, memory, size, gamma):
		if len(memory) >= size:
			states = []
			targets = []

			batch = random.sample(memory, size)

			for experience in batch:
				state, action, next_state, reward, done = experience
				states.append(state)
				q_vals = self.predict(state).tolist()
				if done:
					q_vals[action] = reward
				else: 
					q_vals_next = self.predict(next_state)
					q_vals[action] = reward + gamma * torch.max(q_vals_next).item()
				targets.append(q_vals)
			self.update(states, targets)


class DoubleDQN(DQN):
	
	def __init__(self, state_dim, action_dim, lr):
		super().__init__(state_dim, action_dim, lr)
		self.target = copy.deepcopy(self.nn)


	def target_predict(self, s):
		with torch.no_grad():
			return self.target(torch.Tensor(s))


	def target_update(self):
		self.target.load_state_dict(self.nn.state_dict())


	def replay(self, memory, size, gamma):
		if len(memory) > size:
			states = []
			targets = [] 

			batch = random.sample(memory, size)

			for experience in batch:
				state, action, next_state, reward, done = experience
				states.append(state)

				q_vals = self.predict(state).tolist()
				if done:
					q_vals[action] = reward
				else: 
					q_vals_next = self.target_predict(next_state)
					q_vals[action] = reward + gamma * torch.max(q_vals_next).item()
				targets.append(q_vals)
			self.update(states, targets)

