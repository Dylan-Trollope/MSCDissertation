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
        return loss
        

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
            loss = self.update(states, targets)
            return loss.item()



def train(env, model, episodes, gamma, epsilon, decay, mem_size):
    final_reward = []
    memory = []
    goal_achieved = 0
    episode_num = 0
    average_loss_ep = []
    

    for _ in range(episodes):
        ep_loss = []
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
            loss = model.replay(memory, mem_size, gamma)
            if loss is not None:
                ep_loss.append(loss)
            state = next_state

        epsilon = max(epsilon * decay, 0.01)
        final_reward.append(total)
        if total >= 200:
            goal_achieved += 1
    
        print("Episode number:", episode_num, "Reward:", total)
        
        avg_loss = np.mean(ep_loss)
        
        average_loss_ep.append(avg_loss)
    print(goal_achieved)

    return average_loss_ep


episodes = 200
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
    ale = train(env, model, episodes, gamma, epsilon, decay, 10)

    f = open("loss_graph.csv", "w+")

    for x in ale:
        f.write(str(x) + "\n")

    f.close()
    

