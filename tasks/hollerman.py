#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:16:12 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .trial import Trial, get_itis
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

class Hollerman(Dataset):
    def __init__(self, 
                rew_times=[4, 6, 8],
                rew_probs=None, # defaults to uniform sampling
                jitter=0,
                ntrials=1000,
                ntrials_per_episode=20,
                iti_min=5, iti_p=1/8, iti_max=0, iti_dist='geometric',
                t_padding=0, include_reward=True,
                include_null_input=False):
        self.ntrials = ntrials
        self.rew_times = [int(rt) for rt in rew_times]
        self.nconds = len(self.rew_times)
        self.rew_probs = rew_probs if rew_probs is not None else (np.ones(self.nconds)/self.nconds).tolist()
        self.jitter = jitter
        self.ntrials_per_cond = np.round(np.array(self.rew_probs) * self.ntrials).astype(int)
        self.ntrials_per_episode = ntrials_per_episode
        
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_null_input = include_null_input
        self.make_trials()

        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")
            
    def make_trial(self, cond, iti):
        isi = self.rew_times[cond]
        if self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        return Trial(0, iti, isi, 1, True, 1, self.t_padding, self.include_reward, self.include_null_input)
    
    def make_trials(self, conds=None, ITIs=None):
        if conds is None:
            # create all trials
            self.conds = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.nconds), self.ntrials_per_cond)])
            
            # shuffle trial order
            np.random.shuffle(self.conds)
        else:
            self.conds = conds
        
        # ITI per trial
        self.ITIs = get_itis(self) if ITIs is None else ITIs
        
        # make trials
        self.trials = [self.make_trial(cond, iti) for cond, iti in zip(self.conds, self.ITIs)]
        
        # stack trials to make episodes
        self.episodes = self.make_episode(self.trials, self.ntrials_per_episode)
    
    def make_episode(self, trials, ntrials_per_episode):
        # concatenate multiple trials in each episode
        episodes = []
        for t in np.arange(0, len(trials)-ntrials_per_episode+1, ntrials_per_episode):
            episode = trials[t:(t+ntrials_per_episode)]

            # add episode info
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
