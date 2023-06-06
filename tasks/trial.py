import numpy as np

def get_itis(self):
    # note: we subtract 1 b/c 1 is the min value returned by geometric
    if self.iti_dist == 'geometric':
        itis = np.random.geometric(p=self.iti_p, size=self.ntrials) - 1
    elif self.iti_dist == 'uniform':
        itis = np.random.choice(range(self.iti_max-self.iti_min+1), size=self.ntrials)
    else:
        raise Exception("Unrecognized ITI distribution")
    return self.iti_min + itis

class Trial:
    def __init__(self, cue, iti, isi, reward_size, show_cue, ncues, t_padding, include_reward, include_null_input):
        self.cue = cue
        self.iti = iti
        self.isi = isi
        self.reward_size = reward_size
        self.show_cue = show_cue
        self.ncues = ncues
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_null_input = include_null_input

        self.trial_index_in_episode = None
        self.make()

    def make(self):
        trial = np.zeros((self.iti + self.isi + 1 + self.t_padding, self.ncues + 1))
        if self.show_cue: # encode stimulus
            trial[self.iti, self.cue] = 1.0
        trial[self.iti + self.isi, -1] = self.reward_size
        
        X = trial[:,:-1]
        y = trial[:,-1:]
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
