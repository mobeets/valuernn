#%% imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from train import make_dataloader, train_model, probe_model
from model import ValueRNN
from tasks.example import Example

#%% make trials

E = Example()

#%% make model

hidden_size = 10 # number of hidden neurons
gamma = 0.93 # discount factor
model = ValueRNN(input_size=E.ncues + E.nrewards,
                 output_size=E.nrewards,
                 hidden_size=hidden_size, gamma=gamma)

#%% train model

epochs = 200
batch_size = 12
dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, optimizer=None, epochs=epochs)
plt.figure(figsize=(3,3), dpi=300), plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

E.make_trials() # create new trials for testing
dataloader = make_dataloader(E, batch_size=12)
trials = probe_model(model, dataloader)[1:] # ignore first trial

#%% visualize value and rpe of example trial

plt.figure(figsize=(4,6), dpi=300)
for cue in range(E.ncues):
    trial = next(trial for trial in trials if trial.cue == cue)
    ts = np.arange(len(trial)) - trial.iti + 1
    plt.subplot(2,1,1)
    plt.plot(ts, trial.value, label='cue {}'.format(cue))
    plt.xlim([-2, max(ts)])
    plt.ylabel('value')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(ts[:-1], trial.rpe)
    plt.xlim([-2, max(ts)])
    plt.ylabel('RPE')
    plt.xlabel('time rel. to cue onset')
plt.tight_layout()

#%%
