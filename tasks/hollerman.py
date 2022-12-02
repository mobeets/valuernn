#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:16:12 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            
        isis = [self.rew_times[cond] for cond in range(self.nconds)]
        if len(np.unique(isis)) < self.nconds:
            raise Exception("Bin size is too coarse for provided reward times")
            
    def make_trial(self, cond, iti):
        isi = self.rew_times[cond]
        if self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        trial = np.zeros((iti + isi + 1 + self.t_padding, 1 + 1))
        trial[iti, 0] = 1. # encode stimulus
        trial[iti + isi, -1] = 1. # encode reward
        return trial
    
    def make_trials(self, conds=None, ITIs=None):
        if conds is None:
            # create all trials
            self.conds = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.nconds), self.ntrials_per_cond)])
            
            # shuffle trial order
            np.random.shuffle(self.conds)
        else:
            self.conds = conds
        
        # ITI per trial
        if ITIs is None:
            # note: we subtract 1 b/c 1 is the min value returned by geometric
            if self.iti_dist == 'geometric':
                itis = np.random.geometric(p=self.iti_p, size=self.ntrials) - 1
            elif self.iti_dist == 'uniform':
                itis = np.random.choice(range(self.iti_max-self.iti_min+1), size=self.ntrials)
            else:
                raise Exception("Unrecognized ITI distribution")
            self.ITIs = self.iti_min + itis
        else:
            self.ITIs = ITIs
        
        # make trials
        self.trials = [self.make_trial(cond, iti) for cond, iti in zip(self.conds, self.ITIs)]
        
        # stack trials to make episodes
        self.original_trials = self.trials
        self.trials, self.trial_lengths = self.concatenate_trials(self.trials, self.ntrials_per_episode)
    
    def concatenate_trials(self, trials, ntrials_per_episode):
        # concatenate multiple trials in each episode
        episodes = []
        trial_lengths = []
        # for t in range(len(trials)-ntrials_per_episode+1):
        for t in np.arange(0, len(trials)-ntrials_per_episode+1, ntrials_per_episode):
          ctrials = trials[t:(t+ntrials_per_episode)]
          ctrial_lengths = [len(x) for x in ctrials]
          trial_lengths.append(ctrial_lengths)
          episode = np.vstack(ctrials)
          episodes.append(episode)
        return episodes, trial_lengths
    
    def __getitem__(self, index):
        X = self.trials[index][:,:-1]
        y = self.trials[index][:,-1:]
        trial_lengths = self.trial_lengths[index]
        
        # augment X with previous y
        if self.include_reward:
            X = np.hstack([X, y])
        if self.include_null_input:
            z = (X.sum(axis=1) == 0).astype(np.float)
            X = np.hstack([X, z[:,None]])
            assert np.all(np.sum(X, axis=1) == 1)

        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths)
    
    def __len__(self):
        return len(self.trials)
