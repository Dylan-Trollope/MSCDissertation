import numpy as np 
import time
import torch




def er_dql(env, model, episodes, gamma, epsilon, decay, replay_size):
    final_reward = []
    memory = []
    goal_achieved = 0
    episode_num = 0
    

    for _ in range(episodes):
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
            #env.render()
            total += reward
            memory.append((state, action, next_state, reward, done))
            # if 25 is a list, this doesnt need to be computed again
            model.predict(state).tolist()
            model.replay(memory, replay_size, gamma)
            state = next_state

        epsilon = max(epsilon * decay, 0.01)
        final_reward.append(total)
        if total >= 200:
            goal_achieved += 1
    
        print("Episode number:", episode_num, "Reward:", total)
    print(goal_achieved)
    return final_reward, goal_achieved




def memless_dql(env, model, episodes, gamma, epsilon, decay):
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
            if np.random.random() > epsilon:
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
        print("Episode number:", episode_num, "Reward:", total)

    return final_reward, goal_achieved


def double_dql(env, model, episodes, gamma, epsilon, decay, replay_size, update_freq):
    final_reward = []
    memory = []
    goal_achieved = 0
    episode_num = 0

    for episode in range(episodes):
        episode_num += 1
        if episode % update_freq == 0:
            model.target_update()

        state = env.reset()
        done = False
        total = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total += reward
            memory.append((state, action, next_state, reward, done))
            model.predict(state)
            model.replay(memory, replay_size, gamma)

            state = next_state

        epsilon = max(epsilon * decay, 0.01)
        final_reward.append(total)
        if total >= 200:
            goal_achieved += 1
        print("Episode number:", episode_num, "Reward:", total)
    
    return final_reward, goal_achieved


