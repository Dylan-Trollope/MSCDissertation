import gym 

from DQNAgent import DQNAgent


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
	episode_rewards = []

	for episode in range(max_episodes):
		state = env.reset()
		episode_reward = 0

		for step in range(max_steps):
			action = agent.get_action(state)
			next_state, reward, done, _ = env.step(action)
			agent.replay_buffer.push(state, action, reward, next_state done)
			episode_reward += episode_reward
			
			if len(agent.replay_buffer) > batch_size:
				agent.update(batch_size)

			if done or step == max_steps-1:
				episode_rewards.append(episode_reward)
				print("Episode", episode, "has reward", episode_reward)
				break

			state = next_state

	return episode_rewards
	

env_id = "CartPole-v0"
MAX_EPISODES = 10000
MAX_STEPS = 500
BATCH_SIZE = 32 

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, NAX_STEPS, BATCH_SIZE)
