#%% imports

import numpy as np
from train import make_dataloader
from train_bptt import data_saver, data_saver_to_trials, train_model_TBPTT
from tasks.inference import ValueInference
from model import ValueRNN
import matplotlib.pyplot as plt
import torch

import matplotlib as mpl
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

E = ValueInference(ncues=2, nblocks=2, nblocks_per_episode=50,
    ntrials_per_block=200, cue_probs=None,
    is_trial_level=False, ntrials_per_block_jitter=0,
    reward_probs_per_block={0: (1,0), 1: (0,1)},
    reward_times_per_block=(2,2), iti_p=0.5, iti_min=20,
    nepisodes=1,
    reward_offset_if_trial_level=False)

#%% train model

hidden_size = 50 # number of hidden neurons
epochs = 1
batch_size = 1
window_size = 50 # E.ntrials_per_block * 1
stride_size = 25
gamma = 0.93
auto_readout_lr = 0.0

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size,
                 gamma=gamma if not E.is_trial_level else 0)
model.reset(seed=555)

dataloader = make_dataloader(E, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
scores, training_data, weights = train_model_TBPTT(model, dataloader,
                optimizer=optimizer, epochs=epochs,
                reward_is_offset=not E.is_trial_level,
                data_saver=data_saver, auto_readout_lr=auto_readout_lr,
                window_size=window_size, stride_size=stride_size,
                print_every=50)

loss = np.hstack([y['loss'] for x in training_data for y in x])
plt.plot(loss, '.-'), plt.xlabel('# windows'), plt.ylabel('loss')
training_trials = data_saver_to_trials(E.trials, training_data)

#%% plot value estimates of each cue over time

vs = []
for t, trial in enumerate(training_trials):
    plt.plot(t, trial.value[trial.iti], marker='o' if trial.y.sum() > 0 else 'x', color='b' if trial.cue == 0 else 'r', markersize=5)
    vs.append((trial.block_index_in_episode, trial.block_index, trial.cue, trial.value[trial.iti], trial.y.sum()))
vs = np.array(vs)
plt.xlabel('trial index')
plt.ylabel('value estimate at CS onset')
plt.ylim([-1,2])

#%% plot value over block switches

bis = np.unique(vs[:,0])
r_clrs = plt.cm.Reds(np.linspace(0,1,len(bis)))
b_clrs = plt.cm.Blues(np.linspace(0,1,len(bis)))
for i, bi in enumerate(bis):
    ix = (vs[:,0] == bi) & (vs[:,2] == 0)
    plt.plot(vs[ix,3], color=r_clrs[i] if vs[ix,1][0] == 0 else b_clrs[i])
plt.xlabel('trial index rel. to reversal')
plt.ylabel('value estimate')
