#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:55:18 2022

@author: mobeets
"""
import numpy as np 
import torch
from torch.utils.data import Dataset
from tasks.trial import Trial, get_itis
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

class Babayan(Dataset):
    def __init__(self, nblocks=(1000, 1000),
                ntrials_per_block=(5,5),
                reward_sizes_per_block=(1,3),
                reward_times_per_block=(5,5),
                jitter=0,
                ntrials_per_episode=50,
                iti_min=5, iti_p=1/8, iti_max=0, iti_dist='geometric',
                t_padding=0, include_reward=True,
                include_unique_rewards=False,
                include_null_input=False):
        self.nblocks = nblocks
        self.ntrials_per_block = ntrials_per_block
        assert len(np.unique(ntrials_per_block)) == 1
        self.ntrials = sum([x*y for x,y in zip(nblocks, ntrials_per_block)])
        self.reward_sizes_per_block = reward_sizes_per_block
        self.reward_times_per_block = reward_times_per_block
        self.jitter = jitter
        self.ntrials_per_episode = ntrials_per_episode
        if self.ntrials_per_episode is None:
            self.ntrials_per_episode = self.ntrials
        
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_unique_rewards = include_unique_rewards
        assert not self.include_unique_rewards
        self.include_null_input = include_null_input
        self.make_trials()
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")
            
    def make_trial(self, block_index, iti):
        rew = self.reward_sizes_per_block[block_index]
        isi = self.reward_times_per_block[block_index]
        if self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        trial = Trial(0, iti, isi, rew, True, 1, self.t_padding, self.include_reward, self.include_null_input)
        trial.block_index = block_index
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
        self.ITIs = get_itis(self) if ITIs is None else ITIs
        
        # make trials
        self.trials = [self.make_trial(block_index, iti) for block_index, iti in zip(self.block_indices, self.ITIs)]
        
        # stack trials to make episodes
        self.episodes = self.make_episodes(self.trials, self.ntrials_per_episode)
    
    def make_episodes(self, trials, ntrials_per_episode):
        # concatenate multiple trials in each episode
        episodes = []
        for t in np.arange(0, len(trials)-ntrials_per_episode+1, ntrials_per_episode):
            episode = trials[t:(t+ntrials_per_episode)]

            # add episode info
            ntrials_per_block = self.ntrials_per_block[0]
            for ti, trial in enumerate(episode):
                trial.index_in_episode = ti
                trial.rel_trial_index = (trial.index_in_episode % ntrials_per_block)
                if trial.index_in_episode > ntrials_per_block:
                    trial.prev_block_index = trials[ti-ntrials_per_block].block_index
                else:
                    trial.prev_block_index = None

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

