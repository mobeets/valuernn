import numpy as np
import torch
from torch.utils.data import Dataset
from .trial import Trial
device = torch.device('cpu')

class Example(Dataset):
    def __init__(self, ntrials=1000, ntrials_per_episode=20):
        self.ntrials = ntrials
        self.ntrials_per_episode = ntrials_per_episode
        self.ncues = 2 # number of distinct cues (CS)
        self.nrewards = 1 # number of distinct reward types (US)
        self.make_trials()

    def make_trial(self):
        """ creates a trace conditioning trial with a random delay (isi) """
        iti = 10 # delay before trial onset
        cue = np.random.choice(np.arange(self.ncues)) # CS identity
        isi = 5 if cue == 0 else 9 # reward delay
        return Trial(cue=cue, iti=iti, isi=isi, reward_size=1, show_cue=True, ncues=self.ncues)
    
    def make_trials(self):
        """ creates trials and episodes for a random experiment """

        # make trials
        self.trials = [self.make_trial() for i in range(self.ntrials)]
        
        # concatenate trials to make episodes
        self.episodes = self.make_episodes(self.trials, self.ntrials_per_episode)
    
    def make_episodes(self, trials, ntrials_per_episode):
        """ creates episodes by concatenating trials """

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
        """ returns episode by index """
        episode = self.episodes[index]
        X = np.vstack([trial.X for trial in episode])
        y = np.vstack([trial.y for trial in episode])
        trial_lengths = [len(trial) for trial in episode]
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths, episode)

    def __len__(self):
        """ returns total number of episodes """
        return len(self.episodes)
