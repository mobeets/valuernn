#%% imports

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

from tasks.contingency import Contingency
E = Contingency(mode='garr2023', ntrials=10000, ntrials_per_episode=20)

# from tasks.blocking import Blocking, BlockingTrialLevel
# rew_size_fcn = lambda p: (p, 1-p, 1)
# rew_size_sampler = lambda: rew_size_fcn(np.random.random())
# E = BlockingTrialLevel(rew_size_sampler=rew_size_sampler)

#%% make model

from model import ValueRNN
hidden_size = 50 # number of hidden neurons

# import torch; gamma = torch.Tensor([0.9, 0.95]) # discount rate
gamma = 0.93

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size, gamma=gamma)
model.to('cpu')
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

from train import make_dataloader, train_model
dataloader = make_dataloader(E, batch_size=12)
scores, _, weights = train_model(model, dataloader, lr=0.003, epochs=1000)

#%% plot loss

import matplotlib.pyplot as plt
plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

# model.gamma = model.gamma.numpy()
from train import probe_model
# E.ntrials_per_episode = E.ntrials
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)#[1:]

#%% plot value/rpe in Carr2023

import numpy as np

plt.figure(figsize=(8,3))
for c,trial in enumerate(responses[:20]):
    if trial.y.sum() == 0:
        continue
    plt.subplot(1,E.ncues,trial.cue+1)
    t = trial.iti-2
    # t = 0
    plt.plot(trial.rpe[t:,0], 'r-', label='ND')
    plt.plot(trial.rpe[t:,1], 'b-', label='D')
    plt.plot(np.abs(trial.rpe[t:,1] - trial.rpe[t:,0]), 'k-', label='|D-ND|')
    plt.ylim([-1, 1])
    plt.xticks([1,10], ['cue' if trial.cue < 2 else '', 'reward'])
    plt.xlabel('time')
    plt.ylabel('RPE')
    if trial.cue == 0:
        plt.legend(['food population', 'water population'])
        plt.title('ND cue')
    elif trial.cue == 1:
        plt.title('D cue')
    else:
        plt.title('No cue')
plt.tight_layout()

#%% Fig 7B of Carr et al., 2023

f = lambda trial: np.abs(trial.rpe[:,1] - trial.rpe[:,0])

plt.figure(figsize=(9,3))
for i in range(4):
    plt.subplot(1,4,i+1)

plt.tight_layout()

#%%

cue_indices = [2]*10 + [0]*10 + [1]*10
E.ntrials_per_episode = len(cue_indices)
E.make_trials(cue_indices=cue_indices, rew_size_sampler=lambda: (1,0,1)) # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)#[1:]

# %%

ixf = lambda trial: trial.index_in_episode > 0 and trial.index_in_episode != E.ntrials_per_episode-1
ixf = lambda trial: True

V = [trial.y.sum() for trial in responses if ixf(trial)]
Vhat = np.vstack([trial.value for trial in responses if ixf(trial)])
Z = np.vstack([trial.Z for trial in responses if ixf(trial)])

t = 0
N = 1000
plt.plot(V[t:(t+N)], '.-')
plt.plot(Vhat[t:(t+N)], '.-')

# %%
