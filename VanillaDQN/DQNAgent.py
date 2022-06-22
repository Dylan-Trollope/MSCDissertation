import random
import numpy as np 

import torch
import torch.nn as nn
import torch.autograd as autograd

from DQN import DQN, ConvolutionalDQN
from BasicBuffer import BasicBuffer



class DQNAgent:

	def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000, use_conv=True):
		self.env = env 
		self.learning_rate = learning_rate
		self.gamma = gamma 
		self.replay_buffer = BasicBuffer(max_size=buffer_size)
		
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"

		if use_conv:
			self.model = ConvolutionalDQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
		else:
			self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters())
		self.mse_loss = nn.MSELoss()


	def get_action(self, state, epsilon=0.20):
		if np.random.randn() < epsilon:
			return self.env.action_space.sample()
		else:
			state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
			q_vals = self.model.forward(state)
			return np.argmax(q_vals.cpu().detach().numpy())


	def compute_loss(self, batch):
		states, actions, rewards, next_states, dones = batch
		
		states = torch.FloatTensor(states).to(self.device)
		actions = torch.LongTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		dones = torch.FloatTensor(dones).to(self.device)
	
		current_q = self.model.forward(states).gather(1, actions.unsqueeze(1))
		current_q = current_q.squeeze(1)

		next_q = self.model.forward(next_states)
		max_next_q = torch.max(next_q, 1)[0]
		expected_q = rewards.squeeze(1) + self.gamma * max_next_q
		
		loss = self.mse_loss(current_q, expected_q)
		return loss


	def update(self, batch_size):
		batch = self.replay_buffer.sample(batch_size)
		loss = self.compute_loss(batch)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		

