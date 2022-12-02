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
# from beliefs import discounted_return
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        trial = np.zeros((iti + isi + 1 + self.t_padding, self.ncues + 1))
        trial[iti, cue] = 1.0 # encode stimulus
        if not is_omission:
            trial[iti + isi, -1] = 1.0 # encode reward
        return trial

    def make_trials(self, cues=None, ITIs=None):
        if cues is None:
            self.cues = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.ncues), self.ntrials_per_cue)])
            np.random.shuffle(self.cues)
        else:
            self.cues = cues
        
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
        self.trials = [self.make_trial(cue, iti) for cue, iti in zip(self.cues, self.ITIs)]
        
        # stack trials to make episodes
        self.original_trials = self.trials
        self.trials, self.trial_lengths = self.concatenate_trials(self.trials, self.ntrials_per_episode)

    # def smooth_rewards(self, gamma):
    #     if gamma <= 0:
    #         return
    #     for index in range(len(self.trials)):
    #         self.trials[index][:,-1] = discounted_return(self.trials[index][:,-1], gamma, 1)
    
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
    
    def to_responses(self, constructor=None):
        responses = []
        for i, X in enumerate(self.original_trials):
            y = X[:,-1:]
            cue = np.where(X[:,:-1].sum(axis=0))[0][0]
            iti = np.where(X.sum(axis=1))[0][0]
            if y.sum() > 0:
                isi = np.where(y)[0][0] - iti
            else:
                isi = np.nan
            
            n = X.shape[0]
            data = {'cue': cue, 'iti': iti, 'isi': isi,
                    'X': np.nan * np.ones(n), 'y': np.nan * np.ones(n),
                    'value': np.nan * np.ones(n), 'rpe': np.nan * np.ones(n),
                    'Z': np.nan * np.ones(n), 'index_in_episode': np.nan}
            if constructor is not None:
                data = constructor(**data)
            responses.append(data)
        return responses
    
    def __getitem__(self, index):
        X = self.trials[index][:,:-1]
        y = self.trials[index][:,-1:]
        trial_lengths = self.trial_lengths[index]
        
        # augment X with previous y
        if self.include_reward:
            # yprev = np.vstack([np.zeros((1, 1)), y[:-1,:]])
            # X = np.hstack([X, yprev])
            X = np.hstack([X, y])
        if self.include_null_input:
            z = (X.sum(axis=1) == 0).astype(np.float)
            X = np.hstack([X, z[:,None]])
            assert np.all(np.sum(X, axis=1) == 1)

        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths)

    def __len__(self):
        return len(self.trials)
