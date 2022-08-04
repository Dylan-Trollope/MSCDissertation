import torch
import gym
import random
import numpy as np 

from SingleReplay import DQN

# define the model 

class ER(DQN):

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



def train(env, model, episodes, gamma, epsilon, decay, mem_size):
	final_reward = []
	memory = []
	goal_achieved = 0
	episode_num = 0
	

	for _ in range(episodes):
		episode_num += 1

		state = env.reset()
		done = False
		total = 0

		while not done:
			# should this be converted to a list? 
			q_values = model.predict(state)
			# print(q_values)
			

			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				action = torch.argmax(q_values).item()

			next_state, reward, done, _ = env.step(action)
			#env.render()
			total += reward
			memory.append((state, action, next_state, reward, done))
			# if 25 is a list, this doesnt need to be computed again
			model.predict(state).tolist()
			model.replay(memory, mem_size, gamma)
			state = next_state

		epsilon = max(epsilon * decay, 0.01)
		final_reward.append(total)
		if total >= 200:
			goal_achieved += 1
	
		print("Episode number:", episode_num, "Reward:", total)
	print(goal_achieved)
	return final_reward, goal_achieved


episodes = 50
lr = 0.001

gamma = 0.9
epsilon = 0.3
decay = 0.99
UPDATE = 10


if __name__ == "__main__":

	env = gym.make("CartPole-v1")
	obs_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	model = ER(obs_dim, action_dim, lr)
	train(env, model, episodes, gamma, epsilon, decay, 10)
