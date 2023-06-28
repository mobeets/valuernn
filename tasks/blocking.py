#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 2023

@author: mobeets
"""
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset
from .trial import Trial, get_itis
device = torch.device('cpu')

class Blocking(Dataset):
    def __init__(self,
                 cues=([0], [1], [0,1]),
                 reward_times=(10, 10, 10),
                 rew_probs=(1.0, 1.0, 1.0), # probability of reward on a given trial
                 rew_sizes=(0.35, 0.65, 1),
                 rew_size_sampler=None,
                 ntrials_per_cue=(300, 300, 300),
                 ntrials_per_episode=1,
                 include_reward=True,
                 iti_min=0, iti_p=0.125, iti_max=0, iti_dist='geometric',
                 t_padding=0):
        self.cues = cues
        self.ncues = len(set([x for xs in self.cues for x in xs]))
        self.reward_times = reward_times
        self.rew_sizes = rew_sizes
        self.rew_size_sampler = rew_size_sampler
        self.nrewards = 1 # reward dimensionality (e.g., all rewards are water)
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.rew_probs = rew_probs
        self.ntrials_per_episode = ntrials_per_episode
        if not hasattr(ntrials_per_cue, '__iter__'):
            ntrials_per_cue = ntrials_per_cue*(np.ones(len(self.cues)).astype(int))
        self.ntrials_per_cue = ntrials_per_cue
        self.ntrials = sum(self.ntrials_per_cue)

        self.make_trials(self.rew_size_sampler)
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")

    def make_trial(self, cue_index, iti, rew_sizes):
        is_omission = np.random.rand() > self.rew_probs[cue_index]
        cue = self.cues[cue_index]
        isi = self.reward_times[cue_index]
        rew_size = 0 if is_omission else rew_sizes[cue_index]
        return Trial(cue, iti, isi, rew_size, True, self.ncues, self.t_padding, self.include_reward)

    def make_trials(self, cue_indices=None, ITIs=None, rew_size_sampler=None):
        if cue_indices is None:
            self.cue_indices = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(len(self.cues)), self.ntrials_per_cue)])
            np.random.shuffle(self.cue_indices)
        else:
            self.cue_indices = cue_indices
        
        # ITI per trial
        self.ITIs = get_itis(self) if ITIs is None else ITIs

        # reward sizes per trial
        if rew_size_sampler is not None:
            nepisodes = np.ceil(self.ntrials / self.ntrials_per_episode).astype(int)
            self.rew_sizes_per_episode = [rew_size_sampler() for _ in range(nepisodes)]
            self.rew_sizes_per_trial = [[rew_sizes]*self.ntrials_per_episode for rew_sizes in self.rew_sizes_per_episode]
            self.rew_sizes_per_trial = [x for xs in self.rew_sizes_per_trial for x in xs][:self.ntrials]
        else:
            self.rew_sizes_per_trial = [self.rew_sizes]*self.ntrials
        
        # make trials
        self.trials = [self.make_trial(cue, iti, rew_sizes) for cue, iti, rew_sizes in zip(self.cue_indices, self.ITIs, self.rew_sizes_per_trial)]
        
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

class BlockingTrialLevel(Dataset):
    def __init__(self, ntrials_per_cue=(300, 300, 300),
                 cues=([0], [1], [0,1]),
                 rew_probs=(1.0, 1.0, 1.0), # probability of reward on a given trial
                 rew_sizes=(0.35, 0.65, 1),
                 rew_size_sampler=None,
                 ntrials_per_episode=10,
                 include_reward=True):
        self.cues = cues
        self.ncues = len(set([x for xs in self.cues for x in xs]))
        self.nrewards = 1 # reward dimensionality (e.g., all rewards are water)
        self.include_reward = include_reward
        self.rew_probs = rew_probs
        self.rew_sizes = rew_sizes
        self.rew_size_sampler = rew_size_sampler
        self.ntrials_per_episode = ntrials_per_episode

        if not hasattr(ntrials_per_cue, '__iter__'):
            ntrials_per_cue = ntrials_per_cue*(np.ones(len(self.cues)).astype(int))
        self.ntrials_per_cue = ntrials_per_cue
        self.ntrials = sum(self.ntrials_per_cue)
        self.make_trials(self.rew_size_sampler)

    def make_trial(self, cue_index, rew_sizes):
        is_omission = np.random.rand() > self.rew_probs[cue_index]
        rew_size = 0 if is_omission else rew_sizes[cue_index]
        cue = self.cues[cue_index]
        iti = 0
        isi = 1 # n.b. we correct this in __getitem__
        # if isi==0, we have the problem that the RNN receives the reward it's supposed to predict
        return Trial(cue, iti, isi, rew_size, True, self.ncues, 0, self.include_reward)

    def make_trials(self, cue_indices=None, rew_size_sampler=None):
        if cue_indices is None:
            self.cue_indices = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(len(self.cues)), self.ntrials_per_cue)])
            np.random.shuffle(self.cue_indices)
        else:
            self.cue_indices = cue_indices

        if rew_size_sampler is not None:
            nepisodes = np.ceil(self.ntrials / self.ntrials_per_episode).astype(int)
            self.rew_sizes_per_episode = [rew_size_sampler() for _ in range(nepisodes)]
            self.rew_sizes_per_trial = [[rew_sizes]*self.ntrials_per_episode for rew_sizes in self.rew_sizes_per_episode]
            self.rew_sizes_per_trial = [x for xs in self.rew_sizes_per_trial for x in xs][:self.ntrials]
        else:
            self.rew_sizes_per_trial = [self.rew_sizes]*self.ntrials
        
        # make trials
        self.trials = [self.make_trial(cue, rew_sizes) for cue, rew_sizes in zip(self.cue_indices, self.rew_sizes_per_trial)]
        
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
        if self.ntrials_per_episode > 1:
            # here we sum across adjacent trials so that X(t) contains r(t-1)
            Xc = X[1:-1].reshape(int((len(X)-2)/2), 2, -1).sum(1)
            X = np.vstack([X[:1], Xc, X[-1:]])
            yc = y[1:-1].reshape(int((len(y)-2)/2), 2, -1).sum(1)
            y = np.vstack([y[:1], yc, y[-1:]])
            trial_lengths = [int(len(trial)/2) if i<len(episode)-1 else len(trial) for i,trial in enumerate(episode)]
        else:
            trial_lengths = [len(trial) for trial in episode]
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths, episode)

    def __len__(self):
        return len(self.episodes)
