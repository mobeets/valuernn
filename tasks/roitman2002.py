import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tasks.trial import get_itis

class Roitman2002(gym.Env):
    def __init__(self, reward_amounts, p_coh=0.6,
                 iti_min=0, iti_p=0.5, iti_max=0, iti_dist='geometric'):
        assert len(reward_amounts) == 3 # one reward amount per action
        self.reward_amounts = reward_amounts # Correct, Incorrect, Wait
        self.observation_space = spaces.Discrete(3, start=-1) # Left, Null, Right
        self.action_space = spaces.Discrete(3) # Left, Right, or Wait
        self.p_coh = p_coh # should be in [0.5, 1.0]
        if self.p_coh < 0.5 or self.p_coh > 1.0:
            raise Exception("p_coh must be in [0.5, 1.0]")
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist

    def _get_obs(self):
        """
        returns observation coherent or incoherent with current state
            e.g., if state == 1, coherent observation is obs=1, incoherent is obs=-1
        """
        if self.t < self.iti:
            return 0
        is_coherent = (np.random.rand() <= self.p_coh)
        return self.state if is_coherent else -self.state
    
    def reset(self, seed=None, options=None):
        """
        start new trial
        """
        super().reset(seed=seed)
        self.t = 0
        self.iti = get_itis(self, ntrials=1)[0]
        self.state = 2*int(np.random.rand() > 0.5) - 1 # -1 or 1
        observation = self._get_obs()
        return observation, None
    
    def step(self, action):
        done = action != 2 # trial ends when decision is made
        if action == 2: # wait
            reward = self.reward_amounts[-1]
        elif self.t < self.iti: # action prior to stim onset is treated as incorrect
            reward = self.reward_amounts[1]
        elif (2*action-1) == self.state: # correct decision
            reward = self.reward_amounts[0]
        else: # incorrect decision
            reward = self.reward_amounts[1]
        if action != 2:
            observation = self._get_obs()
        else:
            observation = None # should not be used, because the trial is now over
        self.t += 1
        info = None
        return observation, reward, done, False, info
