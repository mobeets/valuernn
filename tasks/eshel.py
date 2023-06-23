#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:31:01 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .trial import Trial, get_itis
device = torch.device('cpu')
from numpy.random import default_rng

DEFAULT_REW_SIZES = [1, 2, 3, 5, 10, 20]
class RewardDistibution:
    def __init__(self, rew_sizes=DEFAULT_REW_SIZES, rew_probs=None):
        self.rew_sizes = rew_sizes
        if rew_probs is None:
            rew_probs = np.ones(len(self.rew_sizes))/len(self.rew_sizes)
        self.rew_probs = rew_probs / sum(rew_probs)
        assert len(self.rew_sizes) == len(self.rew_probs)
        self.rng = default_rng()

    def sample(self):
        return self.rng.choice(self.rew_sizes, p=self.rew_probs)

class Eshel(Dataset):
    def __init__(self, 
                rew_size_distributions=[RewardDistibution(), RewardDistibution()],
                rew_times=[8, 8],
                cue_shown=[True, False],
                cue_probs=[0.5, 0.5],
                jitter=1,
                ntrials=1000,
                ntrials_per_episode=20,
                iti_min=5, iti_p=1/8, iti_max=0, iti_dist='geometric',
                t_padding=0, include_reward=True,
                include_unique_rewards=True,
                omission_trials_have_duration=True,
                include_null_input=False):
        self.ntrials = ntrials
        self.cue_probs = cue_probs
        self.rew_size_distributions = rew_size_distributions
        self.cue_shown = cue_shown
        self.jitter = jitter
        self.rew_times = rew_times

        self.ncues_shown = sum(self.cue_shown)
        self.ncues = len(self.cue_probs)
        self.nrewards = 1 # reward dimensionality (e.g., all rewards are water)
        self.ntrials_per_cue = np.round(np.array(self.cue_probs) * self.ntrials).astype(int)
        self.ntrials_per_episode = ntrials_per_episode
        
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_unique_rewards = include_unique_rewards
        self.include_null_input = include_null_input
        self.omission_trials_have_duration = omission_trials_have_duration
        self.make_trials()
        if not all(self.cue_shown) and not all([self.cue_shown[i] for i in range(self.cue_shown.index(False)-1)]):
            raise Exception("All hidden cues must be listed last")
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")

    def make_trial(self, cue, iti):
        rew_size = self.rew_size_distributions[cue].sample()
        isi = int(self.rew_times[cue])
        if rew_size == 0 and not self.omission_trials_have_duration:
            isi = 0
        if isi > 0 and self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        assert isi >= 0
        
        # n.b. including input for all cues, even if they're not shown
        return Trial(cue, iti, isi, rew_size, self.cue_shown[cue], self.ncues, self.t_padding, self.include_reward, self.include_null_input)
    
    def make_trials(self, cues=None, ITIs=None):
        if cues is None:
            # create all trials
            self.cues = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.ncues), self.ntrials_per_cue)])
            
            # shuffle trial order
            np.random.shuffle(self.cues)
        else:
            self.cues = cues
        
        # ITI per trial
        self.ITIs = get_itis(self) if ITIs is None else ITIs
        
        # make trials
        self.trials = [self.make_trial(cue, iti) for cue, iti in zip(self.cues, self.ITIs)]
        
        # stack trials to make episodes
        self.episodes = self.make_episodes(self.trials, self.ntrials_per_episode)
    
    def make_episodes(self, trials, ntrials_per_episode):
        # concatenate multiple trials in each episode
        episodes = []
        for t in np.arange(0, len(trials)-ntrials_per_episode+1, ntrials_per_episode):
            episode = trials[t:(t+ntrials_per_episode)]
            for ti, trial in enumerate(episode):
                trial.index_in_episode = ti
            episodes.append(episode)
        return episodes
    
    def __getitem__(self, index):
        episode = self.episodes[index]
        X = np.vstack([trial.X for trial in episode])
        y = np.vstack([trial.y for trial in episode])
        trial_lengths = [len(trial) for trial in episode]
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths, episode)
    
    def __len__(self):
        return len(self.episodes)
