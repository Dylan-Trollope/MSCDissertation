import gym
import time
import random
import numpy as np


def main():

    env = gym.make("Taxi-v3")

    state_space = env.observation_space
    action_space = env.action_space

    state_size = state_space.n
    action_size = action_space.n

    qtable = np.zeros((state_size, action_size))    
        
    learning_rate = 0.1
    discount_rate = 0.8
    epsilon = 1
    decay_rate = 0.005

    num_episodes = 1500
    max_steps = 200 # per episode

    for episode in range(num_episodes):
        state = env.reset()
        print(state)
        done = False

    

        for s in range(max_steps):
            if random.uniform(0,1) < epsilon:
                action = action_space.sample()
                print(action, end=", ")
            else:
                action = np.argmax(qtable[state, :])
                print(action, end=", ")

            new_state, reward, done, info = env.step(action)
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state, action])
            state = new_state

            if done == True:
                print("done")
                break

        epsilon = np.exp(-decay_rate*episode)

    print("Training completed over {} episodes".format(num_episodes)) 


    # WATCH AGENT
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):
        print("TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        time.sleep(0.1)
        print("Score: {}".format(rewards))
        state = new_state

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
