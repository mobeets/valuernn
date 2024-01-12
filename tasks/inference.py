#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:55:18 2022

@author: mobeets
"""
import numpy as np 
import torch
from torch.utils.data import Dataset
from .trial import Trial, get_itis
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

class ValueInference(Dataset):
    def __init__(self,
                nepisodes=1000,
                nblocks=2,
                nblocks_per_episode=4,
                ntrials_per_block=10,
                ntrials_per_block_jitter=1,
                reward_times_per_block=(5,5),
                reward_sizes_per_block=(1,1),
                reward_probs_per_block={0: (0,1), 1: (1,0)}, # {cue: (blk1, blk2)}
                ncues=2,
                cue_probs=(0.5, 0.5),
                jitter=0,
                iti_min=5, iti_p=1/4, iti_max=0, iti_dist='geometric',
                is_trial_level=False,
                t_padding=0, include_reward=True,
                include_null_input=False):
        self.nepisodes = nepisodes
        self.nblocks = nblocks # number of distinct blocks
        self.nblocks_per_episode = nblocks_per_episode
        self.ntrials_per_block = ntrials_per_block
        self.ntrials_per_block_jitter = ntrials_per_block_jitter
        self.reward_times_per_block = reward_times_per_block
        self.reward_sizes_per_block = reward_sizes_per_block
        self.reward_probs_per_block = reward_probs_per_block
        self.jitter = jitter
        

        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding

        self.is_trial_level = is_trial_level # n.b. ignores iti params, and reward_times_per_block
        if self.is_trial_level:
            self.iti_min = 0
            self.iti_max = 0
            self.iti_p = 1
            self.reward_times_per_block = tuple(0 for _ in self.reward_times_per_block)

        self.include_reward = include_reward
        self.include_null_input = include_null_input
        self.ncues = ncues
        self.cue_probs = cue_probs if cue_probs is not None else np.ones(self.ncues)/self.ncues
        self.nrewards = 1 # reward dimensionality (e.g., all rewards are water)
        self.make_trials()
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")

    def make_trial(self, cue, block_index, iti):
        rew_prob = self.reward_probs_per_block[cue][block_index]
        rew_size = self.reward_sizes_per_block[block_index]
        rew = rew_size if np.random.rand() <= rew_prob else 0
        isi = self.reward_times_per_block[block_index]
        if self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        return Trial(cue, iti, isi, rew, True, self.ncues, self.t_padding, self.include_reward, self.include_null_input)
    
    def make_trials(self):
        self.episodes = []
        self.trials = []
        first_blocks = np.random.choice(self.nblocks, size=self.nepisodes)
        for i in range(self.nepisodes):
            trials = []
            first_block = first_blocks[i]
            prev_block_index = None
            r_prev = 0

            for j in range(self.nblocks_per_episode):
                # block order alternates, starting from random first block
                cur_block_index = (first_block + j) % self.nblocks

                # each block has variable number of trials
                ntrials = np.random.choice(self.ntrials_per_block
                 + np.arange(-self.ntrials_per_block_jitter,self.ntrials_per_block_jitter+1), size=1)
                
                # get cues and ITIs for all trials in this block
                ITIs = get_itis(self, ntrials)
                cues = np.random.choice(self.ncues, size=ntrials, p=self.cue_probs)

                # create each trial in block
                for ti, (cue, iti) in enumerate(zip(cues, ITIs)):
                    trial = self.make_trial(cue, cur_block_index, iti)
                    if self.is_trial_level:
                        # need to offset reward by one, both in X and y, since training learns r(t+1)
                        assert trial.trial_length == 1
                        r_cur = trial.y[0,0]
                        if self.include_reward:
                            trial.X[0,-1] = r_prev
                        trial.y[0,0] = r_prev
                        r_prev = r_cur
                    trial.episode_index = i
                    trial.block_index = cur_block_index
                    trial.block_index_in_episode = j
                    trial.rel_trial_index = ti
                    trial.index_in_episode = len(trials)
                    trial.prev_block_index = prev_block_index
                    trials.append(trial)

                prev_block_index = cur_block_index                
            self.trials.extend(trials)
            self.episodes.append(trials)
        self.ntrials = len(self.trials)
    
    def __getitem__(self, index):
        episode = self.episodes[index]
        X = np.vstack([trial.X for trial in episode])
        y = np.vstack([trial.y for trial in episode])
        trial_lengths = [len(trial) for trial in episode]
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths, episode)

    def __len__(self):
        return len(self.episodes)
