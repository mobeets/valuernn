#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:55:18 2022

@author: mobeets
"""
import numpy as np 
import torch
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Babayan(Dataset):
    def __init__(self, nblocks=(1000, 1000),
                ntrials_per_block=(5,5),
                reward_sizes_per_block=(1,3),
                reward_times_per_block=(5,5),
                jitter=0,
                ntrials_per_episode=50,
                iti_min=5, iti_p=1/8, iti_max=0, iti_dist='geometric',
                t_padding=0, include_reward=True,
                include_unique_rewards=True,
                include_null_input=False):
        self.nblocks = nblocks
        self.ntrials_per_block = ntrials_per_block
        self.ntrials = sum([x*y for x,y in zip(nblocks, ntrials_per_block)])
        self.reward_sizes_per_block = reward_sizes_per_block
        self.reward_times_per_block = reward_times_per_block
        self.jitter = jitter
        self.ntrials_per_episode = ntrials_per_episode
        
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_unique_rewards = include_unique_rewards
        self.include_null_input = include_null_input
        self.make_trials()
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")
            
    def make_trial(self, block_index, iti):
        rew = self.reward_sizes_per_block[block_index]
        isi = self.reward_times_per_block[block_index]
        if self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        
        trial = np.zeros((iti + isi + 1 + self.t_padding, 2))
        trial[iti, 0] = 1.0 # encode stimulus
        trial[iti + isi, -1] = rew # encode reward
        return trial
    
    def make_trials(self, block_indices=None, ITIs=None):
        if block_indices is None:
            # create correct number of blocks
            self.block_indices = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(len(self.nblocks)), self.nblocks)])
            
            # shuffle block order
            np.random.shuffle(self.block_indices)
            
            # get block index per trial
            self.block_indices = np.hstack([b*np.ones(self.ntrials_per_block[b]).astype(int) for b in self.block_indices])
        else:
            self.block_indices = block_indices
        
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
        self.trials = [self.make_trial(block_index, iti) for block_index, iti in zip(self.block_indices, self.ITIs)]
        
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
            if self.include_unique_rewards:
                # include one feature per unique reward quantity
                rs = np.unique(self.reward_sizes_per_block)
                y_uniq = np.zeros((y.shape[0], len(rs)))
                for i,r in enumerate(rs):
                    y_uniq[np.where(y == r)[0],i] = 1.
                X = np.hstack([X, y_uniq])
            else:
                X = np.hstack([X, y])
        if self.include_null_input:
            z = (X.sum(axis=1) == 0).astype(np.float)
            X = np.hstack([X, z[:,None]])
            assert np.all(np.sum(X, axis=1) == 1)

        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths)
    
    def __len__(self):
        return len(self.trials)
