import torch
import gym
import random
import numpy as np 

from SingleReplay import DQN
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pandas as pd


# define the model 

class ER(DQN):

    def update(self, state, y):
        y_pred = self.nn(torch.Tensor(state))
        loss = self.loss(y_pred, Variable(torch.Tensor(y)))
        # print(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        
        

    def predict(self, state):
        with torch.no_grad():
            return self.nn(torch.Tensor(state))

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
            y_pred = self.update(states, targets)




def train(env, model, episodes, gamma, epsilon, decay, mem_size):
    final_reward = []
    memory = []
    goal_achieved = 0
    episode_num = 0
    average_loss_ep = []
    s_v_q = []
    

    for _ in range(episodes):
        ep_loss = []
        episode_num += 1

        state = env.reset()

        done = False
        total = 0

        while not done:
            # should this be converted to a list? 
            q_values = model.predict(state)
            

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            # print(next_state)
            # print(action, q_values)
            # print(next_state[2])
            # if -0.05 < state[2] < 0.05:
            #     if -0.5 < state[0] < 0.5:
            #         reward *= 1.2
            #     else:
            #         reward *= 1.1
            # env.render() 

            if -0.10 < state[2] < 0.10:
                reward *= 1.2

            data = (np.round(state[0], 2), np.round(state[2], 2), q_values[0].item(), q_values[1].item())
            s_v_q.append(data)

            #env.render()
            total += reward
            memory.append((state, action, next_state, reward, done))
            

            model.replay(memory, mem_size, gamma)
            state = next_state
        
        epsilon = max(epsilon * decay, 0.01)
        final_reward.append(total)
        if total >= 200:
            goal_achieved += 1
    
        print("Episode number:", episode_num, "Reward:", total)
        
    
    return s_v_q


episodes = 150
lr = 0.0005

gamma = 0.9
epsilon = 0.4
decay = 0.99
UPDATE = 10


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


first = open("RewEngFirst500_angle.csv", "w+")
last = open("RewEngLast500_angle.csv", "w+")

first.write(",".join(["pos", "angle", "q_0", "q_1"]) + "\n")
last.write(",".join(["pos", "angle", "q_0", "q_1"]) + "\n")

T = 100

for t in range(T):
    print(t)
    model = ER(obs_dim, action_dim, lr)
    s_v_q = train(env, model, episodes, gamma, epsilon, decay, 10)

    for el in s_v_q[:500]:
        first.write(",".join([str(e) for e in el]) + "\n")
    for el in s_v_q[-500:]:
        last.write(",".join([str(e) for e in el]) + "\n")


first.close()
last.close()