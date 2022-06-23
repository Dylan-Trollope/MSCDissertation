from DQN import DQN
import gym 
import numpy as np 
import torch

def deep_QLearning(env, model, episodes, gamma, epsilon, decay):
	final_reward = []
	memory = [] 
	
	episode_num = 0

	for episode in range(episodes):
		episode_num +=1

		state = env.reset()
		done = False
		total = 0
		while not done:
			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				q_values = model.predict(state)
				action = torch.argmax(q_values).item()

			next_state, reward, done, _ = env.step(action)
			total += reward

			memory.append((state, action, next_state, reward, done))
			q_values = model.predict(state).tolist()
		

			if done:
				q_values[action] = reward
				model.update(state, q_values)
				break 

			q_values_next = model.predict(next_state)
			q_values[action] = reward + gamma * torch.max(q_values_next).item()
			model.update(state, q_values)

			state = next_state
	
		epsilon = max(epsilon * decay, 0.01)
		final_reward.append(total)
		print(total)


	return final_reward




			
ENV = gym.make("CartPole-v1")
obs_dim = ENV.observation_space.shape[0] # extract from tuple
action_dim = ENV.action_space.n 

dqn = DQN(obs_dim, action_dim)
train = deep_QLearning(ENV, dqn, 500, 0.9, 0.3, 0.99)


