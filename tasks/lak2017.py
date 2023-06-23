import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Lak2017(gym.Env):
    def __init__(self, reward_amounts, sigma=0.0, obs_min=-50, obs_max=50):
        self.obs_min = obs_min
        self.obs_max = obs_max
        assert len(reward_amounts) == 2 # one reward amount per direction
        self.reward_amounts = reward_amounts
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.sigma = sigma

    def _get_obs(self):
        obs = self.state + self.sigma*np.random.randn()
        obs[obs < self.obs_min] = self.obs_min
        obs[obs > self.obs_max] = self.obs_max
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        observation = self._get_obs()
        return observation, None
    
    def step(self, action):
        done = True
        reward = float(self.reward_amounts[0]*(self.state < 0 and action == 0) + self.reward_amounts[1]*(self.state >= 0 and action == 1))
        observation = None # will not be used, because trial is now over
        info = None
        return observation, reward, done, False, info
