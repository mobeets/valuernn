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

E = ValueInference(ncues=2, nblocks=2, nblocks_per_episode=4,
    ntrials_per_block=50, cue_probs=None, seed=123,
    is_trial_level=False, ntrials_per_block_jitter=0,
    reward_probs_per_block={0: (1,0), 1: (0,1)},
    reward_times_per_block=(2,2), iti_p=0.5, iti_min=6,
    nepisodes=1, reward_offset_if_trial_level=False)

#%% train model

mean_trial_length = np.mean([len(x) for x in E.trials])

hidden_size = 50 # number of hidden neurons
epochs = 1
batch_size = 1
# window_size = int(5 * mean_trial_length)
# window_size = int(E.ntrials_per_block * mean_trial_length)
window_size = 50
stride_size = 1
gamma = 0.8
auto_readout_lr = 0.0
print(f'{E.ntrials_per_block=}, {mean_trial_length=}, {window_size=}, {stride_size=}')

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size,
                 gamma=gamma if not E.is_trial_level else 0)
model.reset(seed=555, initialization_gain=1.0)

dataloader = make_dataloader(E, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
scores, training_data, weights = train_model_TBPTT(model, dataloader,
                optimizer=optimizer, epochs=epochs,
                reward_is_offset=not E.is_trial_level,
                data_saver=data_saver, auto_readout_lr=auto_readout_lr,
                window_size=window_size, stride_size=stride_size,
                print_every=200)

loss = np.hstack([y['loss'] for x in training_data for y in x])
plt.plot(loss, '.-'), plt.xlabel('# windows'), plt.ylabel('loss'), plt.show()
training_trials = data_saver_to_trials(E.trials, training_data)

#% plot value estimates of each cue over time

vs = []
for t, trial in enumerate(training_trials):
    plt.plot(t, trial.value[trial.iti], marker='o' if trial.y.sum() > 0 else 'x', color='b' if trial.cue == 0 else 'r', markersize=5)
    vs.append((trial.block_index_in_episode, trial.block_index, trial.rel_trial_index, trial.cue, trial.value[trial.iti], trial.y.sum()))
vs = np.array(vs)
plt.xlabel('trial index')
plt.ylabel('value estimate at CS onset')
plt.ylim([-1,2])
plt.show()

#% plot value over block switches

cue_index = 0 # cue A
bis = np.unique(vs[:,0])
r_clrs = plt.cm.Reds(np.linspace(0,1,len(bis)))
b_clrs = plt.cm.Blues(np.linspace(0,1,len(bis)))
for i, bi in enumerate(bis):
    ix = (vs[:,0] == bi) & (vs[:,3] == cue_index)
    xsc = vs[ix,2]
    plt.plot(xsc, vs[ix,-2], '.-', color=r_clrs[i] if vs[ix,1][0] == 0 else b_clrs[i])
plt.xlabel('trial index rel. to reversal')
plt.ylabel('value estimate, cue A')

#%% train model with multiple window sizes

batch_size = 1
E = ValueInference(ncues=2, nblocks=2, nblocks_per_episode=15,
    ntrials_per_block=50, cue_probs=None, seed=123,
    is_trial_level=False, ntrials_per_block_jitter=0,
    reward_probs_per_block={0: (1,0), 1: (0,1)},
    reward_times_per_block=(2,2), iti_p=0.5, iti_min=6,
    nepisodes=1, reward_offset_if_trial_level=False)
mean_trial_length = np.mean([len(x) for x in E.trials])
dataloader = make_dataloader(E, batch_size=batch_size)

hidden_size = 50 # number of hidden neurons
epochs = 1
stride_size = 1
gamma = 0.9
auto_readout_lr = 0.0
print(f'{E.ntrials_per_block=}, {mean_trial_length=}, {stride_size=}')

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size,
                 gamma=gamma if not E.is_trial_level else 0)

# window_sizes = [1, 50, 100, 250, 500]
window_sizes = [50, 250, 500, 750];# stride_size = 25
seeds = [4,5,6]

Values = {}
for window_size in window_sizes:
    Values[window_size] = []
    for seed in seeds:
        model.reset(seed=seed, initialization_gain=1.0)
        scores, training_data, weights = train_model_TBPTT(model, dataloader,
                        optimizer=None, epochs=epochs,
                        reward_is_offset=not E.is_trial_level,
                        data_saver=data_saver, auto_readout_lr=auto_readout_lr,
                        window_size=window_size, stride_size=stride_size,
                        print_every=500)

        loss = np.hstack([y['loss'] for x in training_data for y in x])
        plt.plot(loss, '.-'), plt.xlabel('# windows'), plt.ylabel('loss'), plt.show()
        training_trials = data_saver_to_trials(E.trials, training_data)

        vs = []
        for t, trial in enumerate(training_trials):
            vs.append((trial.block_index_in_episode, trial.block_index, trial.rel_trial_index, trial.cue, trial.value[trial.iti], trial.y.sum()))
        vs = np.array(vs)
        Values[window_size].append(vs)

#%% plot results

for c, (window_size, vss) in enumerate(Values.items()):
    plt.subplot(len(Values), 1, c+1)
    plt.title(f'{window_size=}')
    for vs in vss:
        bis = np.unique(vs[:,0])[-4:] # last block(s)
        for bi in bis:
            ix = (vs[:,0] == bi)
            ixa = ix & (vs[:,3] == 0)
            ixb = ix & (vs[:,3] == 1)
            ahigh = np.mean(vs[ixa,-1] > 0) > 0.5
            xsc = vs[:,2]; vsc = np.nan * np.zeros_like(xsc)

            # nan-filled
            vsa = vsc.copy(); vsa[ixa] = vs[ixa,4]; xsa = xsc
            vsb = vsc.copy(); vsb[ixb] = vs[ixb,4]; xsb = xsc

            # xsa = vs[ixa,2]; vsa = vs[ixa,4]
            # xsb = vs[ixb,2]; vsb = vs[ixb,4]

            plt.plot(xsa, vsa, 'r-', marker='.' if ahigh else 'x')
            plt.plot(xsb, vsb, 'b-', marker='x' if ahigh else '.')
            plt.ylim([-0.3,1.5])
            break
plt.tight_layout()
