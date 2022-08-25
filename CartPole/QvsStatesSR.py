import numpy as np 
import torch 
import torch.nn as nn
import gym 
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd

from Visualisation import render_averages_plot, render_plot_with_hist
	
# define the model 

class DQN():

	def __init__(self, state_dim, action_dim, lr):
		super(DQN, self).__init__()
		SIZE = 64

		self.nn = nn.Sequential(
				torch.nn.Linear(state_dim, SIZE),
				nn.LeakyReLU(),
				nn.Linear(SIZE, SIZE * 2),
				nn.LeakyReLU(),
				nn.Linear(SIZE * 2, action_dim)
		)

		self.loss = nn.MSELoss()
		self.optimiser = torch.optim.Adam(self.nn.parameters(), lr)


	def update(self, state, y):
		y_pred = self.nn(torch.Tensor(state))
		loss = self.loss(y_pred, Variable(torch.Tensor(y)))
		# print(loss.item())
		self.optimiser.zero_grad()
		loss.backward()
		self.optimiser.step()
		return y_pred, loss.item()
		


	def predict(self, state):
		with torch.no_grad():
			return self.nn(torch.Tensor(state))


def train(env, model, episodes, gamma, epsilon, decay):


	final_reward = []
	goal_achieved = 0	
	episode_num = 0
	
	states_vs_qs = []	 

	for _ in range(episodes):
		episode_num +=1
		state = env.reset()
		done = False
		total = 0
		
		
		while not done:
			q_values = model.predict(state)
			
			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				action = torch.argmax(q_values).item()

			next_state, reward, done, _ = env.step(action)
			
			# state[0] is cart position
			# state[2] is pole angle 
			
			#env.render()
			total += reward

			if done:
				q_values[action] = reward
				y_pred, loss = model.update(state, q_values)
				data = (np.round(state[0], 2), np.round(state[2], 2), q_values[0].item(), q_values[1].item())
				states_vs_qs.append(data)
				
				break 

			q_values_next = model.predict(next_state)
			q_values[action] = reward + gamma * torch.max(q_values_next).item()
			act_q, loss = model.update(state, q_values)

			data = (np.round(state[0], 2), np.round(state[2], 2), q_values[0].item(), q_values[1].item())
			states_vs_qs.append(data)
			
			state = next_state
	
		epsilon = max(epsilon * decay, 0.01)
		final_reward.append(total)
		if total >= 200:
			goal_achieved += 1
	
	
	return states_vs_qs




# parameters
episodes = 150
lr = 0.001

gamma = 0.9
epsilon = 0.4
decay = 0.99
UPDATE = 10


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


first = open("f_ang_ang_speed.csv", "w+")
last = open("l_ang_ang_speed.csv", "w+")

first.write(",".join(["ang speed", "angle", "q_0", "q_1"]) + "\n")
last.write(",".join(["ang speed", "angle", "q_0", "q_1"]) + "\n")
T = 10

for t in range(T):
	print(t)
	model = DQN(obs_dim, action_dim, lr)
	s_v_q = train(env, model, 150, gamma, epsilon, decay)

	for el in s_v_q[:500]:
		first.write(",".join([str(e) for e in el]) + "\n")
	for el in s_v_q[-500:]:
		last.write(",".join([str(e) for e in el]) + "\n")

first.close()
last.close()

