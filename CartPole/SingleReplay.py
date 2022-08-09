import numpy as np 
import torch 
import torch.nn as nn
import gym 
import matplotlib.pyplot as plt
from torch.autograd import Variable


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
        


    def predict(self, state):
        with torch.no_grad():
            return self.nn(torch.Tensor(state))
            
# training loop 


def train(env, model, episodes, gamma, epsilon, decay):


    final_reward = []
    goal_achieved = 0   
    episode_num = 0
    

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
            #env.render()
            total += reward

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
        if total >= 200:
            goal_achieved += 1
        print("Episode number:", episode_num, "Reward:", total, "Epsilon:", epsilon)
        # print("Q values", q_values)
        # print("State", state)
        

    return final_reward, goal_achieved



# parameters
episodes = 150
lr = 0.001

gamma = 0.9
epsilon = 0.3
decay = 0.99
UPDATE = 10

def average(runs, env, episodes):
    all_rewards = []
    successes = []

    for _ in range(runs):
        model = DQN(obs_dim, action_dim, lr)

        rewards, goals = train(env, model, episodes, gamma, epsilon, decay)
        all_rewards.append(rewards)
        successes.append(goals)

    
    rewards_array = np.array([np.array(i) for i in all_rewards])
    return successes, rewards_array


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(obs_dim, action_dim, lr)
    rewards, goals = train(env, model, episodes, gamma, epsilon, decay)

    goals, rewards = average(10, env, episodes)
    
    for i in range(len(rewards)):
        plt.plot(range(episodes), rewards[i], label='_nolegend_')
    plt.axhline(y=200, color='r', linestyle='--', label='goal')
    plt.legend()
    plt.title("CartPole rewards using random strategy over 10 runs")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()






