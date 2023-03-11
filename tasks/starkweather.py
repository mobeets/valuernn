#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:22:16 2022

@author: mobeets
"""
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset
from tasks.trial import Trial, get_itis
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

class Starkweather(Dataset):
    def __init__(self, ncues=1, ntrials_per_cue=300, omission_probability=0.0,
                 include_reward=True,
                 include_null_input=False,
                 omission_trials_have_duration=True,
                 ntrials_per_episode=1,
                 bin_size=0.2,
                 iti_min=0, iti_p=0.5, iti_max=0, iti_dist='geometric',
                 t_padding=0, half_reward_times=False):
        self.ncues = ncues
        self.include_reward = include_reward
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.bin_size = bin_size
        self.t_padding = t_padding
        self.omission_probability = omission_probability
        self.ntrials_per_episode = ntrials_per_episode
        self.ntrials_per_cue = ntrials_per_cue*(np.ones(self.ncues).astype(int))
        self.ntrials = sum(self.ntrials_per_cue)
        self.omission_trials_have_duration = omission_trials_have_duration
        self.include_null_input = include_null_input
        self.half_reward_times = half_reward_times
        self.make_trials()
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")
        # print("WARNING! Temporarily changing reward times for Cue A")

    def get_reward_time(self, cue):
        rts = np.arange(1.2, 3.0, 0.2)
        reward_times = (rts/self.bin_size).astype(int)+1
        if self.half_reward_times:
            rts_A = rts[4:]
            reward_times_A = reward_times[4:]
        else:
            rts_A = rts
            reward_times_A = reward_times
        
        if cue == 0:
            is_omission = np.random.rand() < self.omission_probability
            if is_omission and not self.omission_trials_have_duration:
                return 0, is_omission
            ISIpdf = scipy.stats.norm.pdf(rts_A, rts.mean(), 0.5)
            ISIpdf = ISIpdf/ISIpdf.sum()
            isi = reward_times_A[np.random.choice(len(ISIpdf), p=ISIpdf)].astype(int)
            if is_omission:
                isi = max(reward_times_A)
            return isi, is_omission
        elif cue == 1:
            is_omission = np.random.rand() < self.omission_probability
            if is_omission and not self.omission_trials_have_duration:
                return 0, is_omission
            return reward_times[0], is_omission
        elif cue == 2:
            is_omission = np.random.rand() < self.omission_probability
            if is_omission and not self.omission_trials_have_duration:
                return 0, is_omission
            return reward_times[-1], is_omission
        elif cue == 3:
            return reward_times[0], True
        else:
            raise Exception("No reward time defined for cue {}".format(cue))

    def make_trial(self, cue, iti):
        isi, is_omission = self.get_reward_time(cue)
        rew_size = 0 if is_omission else 1        
        return Trial(cue, iti, isi, rew_size, True, self.ncues, self.t_padding, self.include_reward, self.include_null_input)

    def make_trials(self, cues=None, ITIs=None):
        if cues is None:
            self.cues = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.ncues), self.ntrials_per_cue)])
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
