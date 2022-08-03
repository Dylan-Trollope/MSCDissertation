from Architectures import DQN, ErDQN, DoubleDQN
from Learning import double_dql, memless_dql, er_dql
from RandomStrategy import random_search
from Visualisation import render_plot, render_plot_with_hist
from Experiments import mean_per_episode




import gym 
import sys

# PARAMETERS FOR MODEL 

ENV = gym.make("CartPole-v1")

EPISODES = 50
LEARNING_RATE = 0.001

GAMMA = 0.9
EPSILON = 0.3
DECAY = 0.99
MEMORY_SIZE = 20

UPDATE = 10

OBS_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.n



# PARAMETERS FOR VISUALISATION


if __name__ == "__main__":
	x = range(EPISODES)
	runs = 10
	if sys.argv[1] == "Random":
		if sys.argv[2] == "1":
			alg = lambda _ : random_search(ENV, EPISODES)
			x, y, count = mean_per_episode(None, alg, EPISODES, runs)
			average_goal_achieved = sum(count) / runs
			print(average_goal_achieved)
			render_plot(x, y, "CartPole average using random search over 10 runs for " + str(EPISODES) + " episodes")
		elif sys.argv[2] == '2':
			y, count = random_search(ENV, EPISODES)
			render_plot_with_hist(x, y, count, "CartPole with goal achieved counts", True)


	if sys.argv[1] == "Memless":
		model = DQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
		print(list(model.parameters()))
		if sys.argv[2] == '1':
			alg = lambda model: memless_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY)
			x, y, count = mean_per_episode(model, alg, EPISODES, runs)
			average_goal_achieved = sum(count) / runs
			print(average_goal_achieved)
			render_plot(x, y, "CartPole Performance using DQN without memory")
		elif sys.argv[2] == '2':
			y, count = memless_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY)
			render_plot_with_hist(x, y, count, "CartPole without memory hist", True)
			


	if sys.argv[1] == "ER":
		model = ErDQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
		if sys.argv[2] == '1':
			alg = lambda model: er_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE)
			x, y, count = mean_per_episode(model, alg, EPISODES, runs)
			render_plot(x, y, "CartPole with Experience Replay")



	if sys.argv[1] == "Double":
		model = DoubleDQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
		y, count = double_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE, 10)
		


