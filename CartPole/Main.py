from Architectures import *
from Learning import *
from RandomStrategy import random_search
from Visualisation import render_plot


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

obs_dim = ENV.observation_space.shape[0]
action_dim = ENV.action_space.n

# PARAMETERS FOR VISUALISATION

x = range(EPISODES)

# MODEL = DQN(obs_dim, action_dim)
# MODEL = ER_DQN(obs_dim, action_dim, LEARNING_RATE)
# MODEL = Double_DQN(obs_dim, action_dim, LEARNING_RATE)


#y = memless_DQL(ENV, MODEL, EPISODES, GAMMA, EPSILON, DECAY)
#y = ER_DQL(ENV, MODEL, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE)
#y = double_DQL(ENV, MODEL, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE, UPDATE)

# render_plot(x, y, "CartPole performance using Double Deep Q Learning with Experience Replay", True)


if __name__ == "__main__":
    EPISODES = 50
    x = range(EPISODES)
    if sys.argv[1] == "Random":
        y = random_search(ENV, EPISODES)
        render_plot(x, y, "CartPole performance using random strategy", False)

    if sys.argv[1] == "Memless":
        MODEL = DQN(obs_dim, action_dim, LEARNING_RATE)
        y = memless_DQL(ENV, MODEL, EPISODES, GAMMA, EPSILON, DECAY)
        render_plot(x, y, "CartPole Performance using DQN without memory", True)

    if sys.argv[1] == "ER":
        MODEL = ER_DQN(obs_dim, action_dim, LEARNING_RATE)
        y, count = ER_DQL(ENV, MODEL, EPISODES, GAMMA, EPSILON, DECAY, MEMORY_SIZE)
        render_plot(x, y, count, "CartPole Performance using DQN with Experience Replay", True)



