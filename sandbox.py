#%% imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from train import make_dataloader, train_model, probe_model, train_model_step_by_step
from model import ValueRNN

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

# from tasks.eshel import Eshel
# from tasks.starkweather import Starkweather
# from tasks.trial import RewardAmountDistibution, RewardTimingDistribution

# E = Starkweather(ncues=4, ntrials_per_cue=10000, ntrials_per_episode=20,
#                  iti_min=5, iti_p=0.25)

# rew_times =   [  7, 11]
# cue_probs_1 = [0.5, 0.5]
# E = Eshel(
#     rew_size_distributions=[RewardAmountDistibution([1])]*len(rew_times),
#     rew_time_distibutions=[RewardTimingDistribution([r]) for r in rew_times],
#     cue_shown=[True]*len(rew_times),
#     cue_probs=cue_probs_1,
#     iti_p=0.5,
#     jitter=0,
#     ntrials=1000,
#     ntrials_per_episode=20)

# from tasks.blocking import Blocking, BlockingTrialLevel
# rew_size_fcn = lambda p: (p, 1-p, 1)
# rew_size_sampler = lambda: rew_size_fcn(np.random.random())
# E = BlockingTrialLevel(rew_size_sampler=rew_size_sampler)

#%% make trials

from tasks.inference import ValueInference

corrSign = 'anti'
anticorr_reward_probs = {0: (0,1), 1: (1,0)}
poscorr_reward_probs = {0: (0,1), 1: (0,1)}
reward_probs = anticorr_reward_probs if corrSign == 'anti' else poscorr_reward_probs
E = ValueInference(nblocks_per_episode=4, ntrials_per_block=8,
                is_trial_level=True, ntrials_per_block_jitter=3,
                reward_probs_per_block=reward_probs)

#%% synapse model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from train import make_dataloader, train_model, probe_model, train_model_step_by_step
from model import ValueRNN
from tasks.inference import ValueInference

corrSign = 'anti'
anticorr_reward_probs = {0: (0,1), 1: (1,0)}
poscorr_reward_probs = {0: (0,1), 1: (0,1)}
reward_probs = anticorr_reward_probs if corrSign == 'anti' else poscorr_reward_probs

rnn_is_synapses = True
E = ValueInference(nblocks_per_episode=4, ntrials_per_block=12,
                is_trial_level=True, ntrials_per_block_jitter=3,
                reward_probs_per_block=reward_probs, reward_offset_if_trial_level=not rnn_is_synapses)

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=E.ncues+1, gamma=0 if E.is_trial_level else 0.93,
                 rnn_is_synapses=rnn_is_synapses, bias=False, synapse_count=10)

model.representation.weight.data *= 0
for i in range(model.representation.weight.shape[1]):
    model.representation.weight.data[i,i] = 1
model.representation.weight.requires_grad = False
model.representation.bias.data *= 0
model.representation.bias.requires_grad = False

dataloader = make_dataloader(E, batch_size=12)
scores, other_scores, weights = train_model(model, dataloader, optimizer=None, epochs=100)
plt.plot(scores)

#%% probe synapse model

E.nblocks_per_episode = 1000; E.nepisodes = 1
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)#[1:]

data = np.vstack([(trial.block_index, trial.cue, trial.y.sum(), trial.value[0,0]) for trial in responses])
z = np.vstack([trial.Z for trial in responses])
z = np.vstack([model.initial_state.detach().numpy(), z])
W = model.value.weight.data.numpy()
w = z @ W.T + model.value.bias.data.numpy()

T=24
h = plt.plot(w[:T,:2], '.-', alpha=0.3)
print(data[:T])

#%% make model

hidden_size = 10 # number of hidden neurons

# import torch; gamma = torch.Tensor([0.9, 0.95]) # discount rate
gamma = 0.93 if not E.is_trial_level else 0

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size, gamma=gamma)
model.to('cpu')
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model step-by-step

dataloader = make_dataloader(E, batch_size=1)
scores, other_scores, weights = train_model_step_by_step(model, dataloader, epochs=5, print_every=10)

#%% train model

epochs = 200
batch_size = 12
dataloader = make_dataloader(E, batch_size=batch_size)
optimizer = torch.optim.Adam(
    [
        {'params': model.rnn.parameters(), 'lr': 0.001},
        {'params': model.value.parameters(), 'lr': 0.003},
        {'params': model.bias, 'lr': 0.003},
    ], lr=0.003, amsgrad=False)

optimizer = None

scores, other_scores, weights = train_model(model, dataloader, optimizer=optimizer, epochs=epochs)

#%% plot loss

plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

# model.gamma = model.gamma.numpy()
# E.ntrials_per_episode = E.ntrials
E.nblocks_per_episode = 1000; E.nepisodes = 1
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)#[1:]

#%% plot avg value per cue, per block, over time

tPre = E.iti_min # time steps to show before cue onset
plt.figure(figsize=(6,3))
for bi in [0,1]:
    plt.subplot(1,2,bi+1)
    for cue in range(E.ncues):
        trial_keeper = lambda trial: trial.block_index == bi and trial.rel_trial_index > 0 and trial.prev_block_index != bi and trial.prev_block_index is not None
        values = [trial.value[trial.iti-tPre:] for trial in responses if trial.cue == cue and trial_keeper(trial)]
        if not len(values):
            continue
        try:
            V = np.hstack(values)
        except:
            continue
        vs = V.mean(axis=1)
        xs = np.arange(len(vs))-tPre
        plt.plot(xs, vs, '.-', label='cue {}'.format('A' if cue == 0 else 'B'))
    plt.xlabel('time rel. to odor onset')
    plt.ylabel('$\widehat{V}(x)$')
    plt.title('Block {}'.format(bi+1))
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% plot value over trials as shown in experiment

plt.figure(figsize=(6,3))
clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, trial in enumerate(responses[:50]):
    v = trial.value[trial.iti]
    rewarded = trial.y.sum() > 0 if not E.is_trial_level else (responses[i+1].y.sum() > 0 if E.reward_offset_if_trial_level else responses[i].y.sum())
    plt.plot(i, v, 'o' if rewarded else 'x', color=clrs[trial.cue], markersize=5)
    if trial.rel_trial_index == 0:
        plt.plot((i-0.5)*np.ones(2), [0,1], '-', color=clrs[-trial.block_index+1], alpha=0.2, zorder=-1)
plt.xlabel('trial index')
plt.ylabel('$\widehat{V}(x)$')

#%% plot avg value per cue for each trial in block

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

V = {}
for bi in [0,1]:
    for cue in range(E.ncues):
        V[(bi,cue)] = []
        for t in range(E.ntrials_per_block):
            trial_keeper = lambda trial: trial.block_index == bi and trial.cue == cue and trial.rel_trial_index == t and trial.prev_block_index != bi and trial.prev_block_index is not None
            values = [trial.value[trial.iti] for trial in responses if trial_keeper(trial)]
            V[(bi,cue)].append(np.mean(values))
        lbl = 'block {}, cue {}'.format(bi+1, 'A' if cue == 0 else 'B')
        ys = V[(bi,cue)]
        xs = np.arange(len(ys))
        plt.plot(xs, ys, '-' if bi==0 else '--', color=clrs[cue], label=lbl)
plt.xlabel('trials rel. to reversal')
plt.ylabel('$\widehat{V}(x)$')
plt.xticks(range(0, E.ntrials_per_block, 2))
plt.legend(fontsize=8)
plt.show()

#%% get avg rep per cue, per block

dotfcn = lambda U: np.dot(U[0]/np.linalg.norm(U[0]), U[1]/np.linalg.norm(U[1]))

Z = {}
zsBase = np.vstack([trial.Z[trial.iti-1,:] for trial in responses]).mean(axis=0)
for bi in [0,1]:
    for cue in range(E.ncues):
        Z[(bi,cue)] = []
        for t in range(E.ntrials_per_block):
            trial_keeper = lambda trial: trial.block_index == bi and trial.cue == cue and trial.rel_trial_index == t and trial.prev_block_index != bi and trial.prev_block_index is not None
            zs = [trial.Z[trial.iti,:] for trial in responses if trial_keeper(trial)]
            Z[(bi,cue)].append(np.mean(zs, axis=0) - 1*zsBase)
        Z[(bi,cue)] = np.vstack(Z[(bi,cue)])

ys1 = [dotfcn([z1, z2]) for z1, z2 in zip(Z[(0,0)], Z[(0,1)])]
ys2 = [dotfcn([z1, z2]) for z1, z2 in zip(Z[(1,0)], Z[(1,1)])]
xs = np.arange(len(ys1))
plt.plot(xs, ys1, 'k-', label='block 1')
plt.plot(xs, ys2, 'k--', label='block 2')
plt.xlabel('trials rel. to reversal')
plt.ylabel('cosine similarity($z_A - z_0, z_B - z_0$)')
plt.xticks(range(0, E.ntrials_per_block, 2))
plt.legend(fontsize=8)
plt.show()

#%% apply PCA

import numpy as np
from sklearn.decomposition import PCA

def fit_pca(trials):
    Z = np.vstack([trial.Z for trial in trials])
    pca = PCA(n_components=Z.shape[1])
    pca.fit(Z)
    return pca

def apply_pca(trials, pca):
    for trial in trials:
        trial.Z_pc = pca.transform(trial.Z)
    return trials

pca = fit_pca(responses)
apply_pca(responses, pca)

#%% visualize PCs

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

trials = responses[20:]
Z = np.vstack([trial.Z_pc for trial in trials])
X = np.hstack([trial.cue for trial in trials])
B = np.hstack([trial.block_index for trial in trials])

t1 = 0; t2 = 60
Z = Z[t1:t2]
X = X[t1:t2]
B = B[t1:t2]
plt.plot(Z[:,0], Z[:,1], '-')
for z,x,b in zip(Z,X,B):
    plt.plot(z[0], z[1], 'o' if b==0 else 'x', markersize=5, color=clrs[x])

#%%
