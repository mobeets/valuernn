import numpy as np
from numpy.random import default_rng

def get_itis(self, ntrials=None):
    ntrials = ntrials if ntrials is not None else self.ntrials
    
    # note: we subtract 1 b/c 1 is the min value returned by geometric
    if self.iti_dist == 'geometric':
        itis = np.random.geometric(p=self.iti_p, size=ntrials) - 1
    elif self.iti_dist == 'uniform':
        itis = np.random.choice(range(self.iti_max-self.iti_min+1), size=ntrials)
    else:
        raise Exception("Unrecognized ITI distribution")
    return self.iti_min + itis

class RewardAmountDistibution:
    def __init__(self, rew_sizes, rew_probs=None):
        self.rew_sizes = rew_sizes
        if rew_probs is None:
            rew_probs = np.ones(len(self.rew_sizes))/len(self.rew_sizes)
        self.rew_probs = np.array(rew_probs) / sum(rew_probs)
        assert len(self.rew_sizes) == len(self.rew_probs)
        self.rng = default_rng()

    def sample(self):
        return self.rng.choice(self.rew_sizes, p=self.rew_probs)

class RewardTimingDistribution:
    def __init__(self, rew_times, time_probs=None):
        self.rew_times = rew_times
        if time_probs is None:
            time_probs = np.ones(len(self.rew_times))/len(self.rew_times)
        self.time_probs = np.array(time_probs) / sum(time_probs)
        assert len(self.time_probs) == len(self.rew_times)
        self.rng = default_rng()

    def sample(self):
        return self.rng.choice(self.rew_times, p=self.time_probs)

class Trial:
    def __init__(self, cue, iti, isi, reward_size, show_cue, ncues, t_padding=0, include_reward=True, include_null_input=False, do_trace_conditioning=True):
        self.cue = cue
        self.iti = iti
        self.isi = isi
        self.reward_size = reward_size
        self.show_cue = show_cue
        self.ncues = ncues
        self.nrewards = len(self.reward_size) if hasattr(self.reward_size, '__iter__') else 1
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_null_input = include_null_input
        self.do_trace_conditioning = do_trace_conditioning
 
        self.trial_index_in_episode = None
        self.make()

    def make(self):
        trial = np.zeros((self.iti + self.isi + 1 + self.t_padding, self.ncues + self.nrewards))
        if self.show_cue: # encode stimulus
            if self.do_trace_conditioning:
                trial[self.iti, self.cue] = 1.0
            else: # delay conditioning
                trial[self.iti:(self.iti+self.isi+1), self.cue] = 1.0
        trial[self.iti + self.isi, -self.nrewards:] = self.reward_size
        
        X = trial[:,:-self.nrewards]
        y = trial[:,-self.nrewards:]
        if self.include_reward:
            X = np.hstack([X, y])
        if self.include_null_input:
            z = (X.sum(axis=1) == 0).astype(np.float)
            X = np.hstack([X, z[:,None]])
            assert np.all(np.sum(X, axis=1) == 1)

        self.trial = trial
        self.trial_length = len(trial)
        self.X = X
        self.y = y

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self):
        return self.trial_length

    def __str__(self):
        return f'{self.cue=}, {self.iti=}, {self.isi=}, {self.reward_size=}, {self.index_in_episode=}, {self.trial_length=}'
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.__str__()})'
