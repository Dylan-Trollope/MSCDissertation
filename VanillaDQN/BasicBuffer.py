from collections import deque
import random
import numpy as np

class BasicBuffer:

	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	
	def push(self, state, action, reward, next_state, done):
		experience = (state, action, np.array([reward]), next_state, done)
		self.buffer.append(experience)

	def sample(self, bath_size):
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		done_batch = []

		batch = random.sample(self.buffer, batch_size)




	def sample_sequence():
		pass


	def __len__():
		pass




