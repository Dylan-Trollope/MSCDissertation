from Architectures import DQN, ErDQN, DoubleDQN
from Learning import double_dql, memless_dql, er_dql
from RandomStrategy import random_search
from Visualisation import render_plot
from Experiments import mean_per_episode




import gym 
import sys

# PARAMETERS FOR MODEL 

ENV = gym.make("CartPole-v1")

EPISODES = 100
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
    if sys.argv[1] == "Random":
        if sys.argv[2] == "1":
            print("here")
            x, y, count = mean_per_episode(None, random_search, 10, EPISODES, ENV)
            render_plot(x, y, "CartPole using random search over 10 runs")
        
        #render_plot(x, y, count, "CartPole performance using random strategy", False)

    if sys.argv[1] == "Memless":
        model = DQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
        y, count = memless_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY)
        render_plot(x, y, count, "CartPole Performance using DQN without memory", True)

    if sys.argv[1] == "ER":
        model = ErDQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
        y, count = er_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE)
        render_plot(x, y, count, "CartPole Performance using DQN with Experience Replay", True)

    if sys.argv[1] == "Double":
        model = DoubleDQN(OBS_DIM, ACTION_DIM, LEARNING_RATE)
        y, count = double_dql(ENV, model, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE, 10)
        


