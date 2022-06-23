import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from Visualisation import render_plot

ENV = gym.make("CartPole-v1")
EPISODES = 200

def random_search(env, episodes):
	reward_list = []
	for episode in range(episodes):
		state = env.reset()
		done = False
		total = 0

		while not done:
			action = env.action_space.sample()
			next_state, reward, done, _ = env.step(action)
#			env.render()
			total += reward

			if done:
				break

		reward_list.append(total)
	return reward_list

x = range(EPISODES)
y = random_search(ENV, EPISODES)

render_plot(x,y, "Performance of CartPole using random strategy", "Number of Episodes", "Reward", True)
