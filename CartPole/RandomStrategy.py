import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from Visualisation import render_plot

ENV = gym.make("CartPole-v1")
EPISODES = 200

def random_search(env, episodes):
    reward_list = []
    goal_achieved = 0
    
    for _ in range(episodes):
        
        state = env.reset()
        done = False
        total = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
#           env.render()
            total += reward

            if done:
                break

        reward_list.append(total)
        if total >= 200:
            goal_achieved += 1
    return reward_list, goal_achieved


def averages(runs, env, episodes):
    all_rewards = []
    successes = []

    for _ in range(runs):
        rewards, goals = random_search(env, episodes)
        all_rewards.append(rewards)
        successes.append(goals)

    
    rewards_array = np.array([np.array(i) for i in all_rewards])
    return successes, rewards_array
    



if __name__ == "__main__":








