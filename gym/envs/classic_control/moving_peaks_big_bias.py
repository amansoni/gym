"""
Classic moving peaks benchmark by Jurgen Branke
TODO: Links and references
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MovingPeaksBigBiasEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self):
		self.bias = 100  # bias added to reward based on previous action
		self.tau = 0.02  # seconds between state updates - needed?
		self.center = 5
		self.height = 30
		self.width = 2
		self.low = np.array(-self.center)
		self.high = np.array(self.center)

		self.action_space = range(-10, 11)
		self.observation_space = spaces.Box(self.low, self.high)

		self._seed()
		self.viewer = None
		self.state = None

		self.steps_beyond_done = None

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		assert action in self.action_space, "%r (%s) invalid. valid actions %s"%(action, type(action), self.action_space)
		state = self.state
		height, width, center = state
		self.state = (height, width, -center)
		done = False

		if not done:
			if self.previous_action is None or self.previous_action >= 0:
				b = self.bias
			else:
				b = -self.bias
			reward = height - width * abs(action - center) + b
		elif self.steps_beyond_done is None:
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		self.previous_action = action
		self.current_step = self.current_step + 1
	#print("State: {} reward:{}").format(self.state, reward)

		return np.array(self.state), reward, done, {}

	def _reset(self):
		self.current_step = 0
		self.previous_action = None
		self.state = (self.height, self.width, self.center)
		self.steps_beyond_done = None
		#print("reset")
		return np.array(self.state)

	def _render(self, mode='human', close=False):
		pass

