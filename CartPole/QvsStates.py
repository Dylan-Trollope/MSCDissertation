import numpy as np 
import torch 
import torch.nn as nn
import gym 
import matplotlib.pyplot as plt
from torch.autograd import Variable

from Visualisation import render_averages_plot, render_plot_with_hist

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
        return y_pred, loss.item()
        


    def predict(self, state):
        with torch.no_grad():
            return self.nn(torch.Tensor(state))


def train(env, model, episodes, gamma, epsilon, decay):


    final_reward = []
    goal_achieved = 0   
    episode_num = 0
    average_loss_per_ep = []
    act_q_0 = []
    act_q_1 = []
    pred_q_0 = []
    pred_q_1 = []
    failures = []
    cart_position = []
    pole_angle = []
    q_list = []
    
    
    

    for _ in range(episodes):
        loss_ep = []
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
            
            # state[0] is cart position
            # state[2] is pole angle 
            cart_position.append(state[0])
            pole_angle.append(state[2])
            
            #env.render()
            total += reward

            if done:
                q_values[action] = reward
                y_pred, loss = model.update(state, q_values)
                loss_ep.append(loss)
                act_q_0.append(q_values[0].item())
                act_q_1.append(q_values[1].item())
                q_list += [q_values]
                #q_s = torch.stack((q_s, q_values), dim=0)
                
                pred_q_0.append(y_pred[0].item())
                pred_q_1.append(y_pred[1].item())
                
                break 

            q_values_next = model.predict(next_state)
            # print(q_values_next)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            q_list += [q_values]
            #q_s = torch.stack((q_s, q_values), dim=0)
            # print(q_values)
            act_q, loss = model.update(state, q_values)
            
            
            # print("Next", q_values_next)
            
            
            act_q_0.append(act_q[0].item())
            act_q_1.append(act_q[1].item())
            
            pred_q_0.append(q_values[0].item())
            pred_q_1.append(q_values[1].item())
            
            loss_ep.append(loss)
            state = next_state
    
        epsilon = max(epsilon * decay, 0.01)
        final_reward.append(total)
        avg_loss = np.mean(loss_ep)
        average_loss_per_ep.append(avg_loss)
        if total >= 200:
            goal_achieved += 1
        # print("Episode number:", episode_num, "Reward:", total, "Epsilon:", epsilon)
        # print("Q values", q_values)
        # print("State", state)
        
 
    
    # return final_reward, goal_achieved
    # return final_reward, goal_achieved, average_loss_per_ep
    q_s = torch.stack(tuple([q for q in q_list]))
    # max_qs = [max(q).item() for q in q_s]
    
    
    return q_s, cart_position, pole_angle
    # return final_reward, goal_achieved, average_loss_per_ep, act_q_0, pred_q_0, act_q_1, pred_q_1





# parameters
episodes = 10
lr = 0.001

gamma = 0.9
epsilon = 0.3
decay = 0.99
UPDATE = 10


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = DQN(obs_dim, action_dim, lr)


q, pos, angle = train(env, model, 100, gamma, epsilon, decay)


#8.5
#8.3
q_max = torch.max(q, axis=1)
print(len(q_max[0]))
print(q_max[0].shape)
print(q_max[0].shape)

numpy_q = q_max[0].numpy()
print(numpy_q)

from matplotlib import cm
from scipy.interpolate import griddata


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')



x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
y_min, y_max = env.observation_space.low[2], env.observation_space.high[2]

x_space = np.linspace(x_min, x_max, len(pos))
y_space = np.linspace(y_min, y_max, len(angle))
X, Y = np.meshgrid(x_space, y_space)

Z = griddata((pos, angle), , (X, Y), method='linear')
ax.plot_surface(X, Y, Z)


plt.show()





