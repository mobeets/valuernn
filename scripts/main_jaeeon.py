#%% imports

import numpy as np
from train import make_dataloader, train_model, probe_model, train_model_TBPTT
from tasks.inference import ValueInference
from model import ValueRNN
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

import matplotlib as mpl
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

mode = 'anticorr'
# mode = 'corr'

anticorr_reward_probs = {0: (0,1), 1: (1,0)} # { block_index: (cueAreward, cueBreward), ...}
poscorr_reward_probs = {0: (0,1), 1: (0,1)}
reward_probs = anticorr_reward_probs if mode == 'anticorr' else poscorr_reward_probs
E = ValueInference(nblocks_per_episode=120,
    ntrials_per_block=50,
    is_trial_level=False, ntrials_per_block_jitter=0,
    reward_probs_per_block=reward_probs,
    reward_times_per_block=(1,1), iti_p=0.8,
    nepisodes=1,
    reward_offset_if_trial_level=False)

#%%

E = ValueInference(ncues=1, nblocks=2, nblocks_per_episode=1,
    ntrials_per_block=50, cue_probs=None,
    is_trial_level=False, ntrials_per_block_jitter=0,
    reward_probs_per_block={0: (1,0), 1: (0,1)},
    reward_times_per_block=(5,5), iti_p=0.5, iti_min=20,
    nepisodes=1,
    reward_offset_if_trial_level=False)

#%% prepare to get Trial objects during training

def data_saver(X, y, V_hat, V_target, hs, loss, model, optimizer):
    return {
        'X': X,
        'Z': hs.detach().numpy(),
        'V_hat': V_hat.detach().squeeze(),
        'rpe': V_target.detach().squeeze() - V_hat.detach().squeeze(),
        'loss': loss.item(),
        # 'weights': deepcopy(model.state_dict()),
        # 'optimizer': deepcopy(optimizer.state_dict()['state']),
        }

def data_saver_to_trials(training_data, epoch_index=0):
    batch_size == training_data[epoch_index][0]['X'].shape[1]
    if batch_size > 1:
        raise Exception("You must use batch_size==1.")
    X = np.vstack([entry['X'][:,0,:] for entry in training_data[epoch_index]])
    V = np.hstack([entry['V_hat'] for entry in training_data[epoch_index]])
    Z = np.vstack([entry['Z'][:,0,:] for entry in training_data[epoch_index]])
    rpe = np.hstack([entry['rpe'] for entry in training_data[epoch_index]])

    trials = []
    t_start = 0
    for i, trial in enumerate(E.trials):
        trial = deepcopy(trial)
        t_end = t_start + len(trial)
        Xc = X[t_start:t_end]
        Zc = Z[t_start:t_end]
        Vc = V[t_start:t_end]
        rpec = rpe[t_start:t_end]
        t_pad = len(trial) - len(Xc)
        assert (trial.X[:len(Xc)] == Xc[:len(Xc)]).all()
        if t_pad > 0:
            # some trials will not be processed in full because of stride size
            Xc = np.vstack([Xc, np.nan * np.ones((t_pad, Xc.shape[1]))])
            Zc = np.vstack([Zc, np.nan * np.ones((t_pad, Zc.shape[1]))])
            Vc = np.hstack([Vc, np.nan * np.ones((t_pad,))])
            rpec = np.hstack([rpec, np.nan * np.ones((t_pad,))])
        trial.Z = Zc
        trial.value = Vc
        trial.rpe = rpec
        # trial.readout = W # todo
        trials.append(trial)
        t_start += len(trial)
    return trials

#%% train model

hidden_size = 5 # number of hidden neurons
epochs = 1
batch_size = 1
window_size = 200
stride_size = 5
# gamma = 0.93
gamma = 0.8

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size,
                 gamma=gamma if not E.is_trial_level else 0)
model.reset(seed=555)

dataloader = make_dataloader(E, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
scores, training_data, weights = train_model_TBPTT(model, dataloader,
                optimizer=optimizer, epochs=epochs,
                reward_is_offset=not E.is_trial_level,
                data_saver=data_saver,
                window_size=window_size, stride_size=stride_size,
                print_every=50)

loss = np.hstack([y['loss'] for x in training_data for y in x])
plt.plot(loss), plt.xlabel('# windows'), plt.ylabel('loss')
training_trials = data_saver_to_trials(training_data)

#%% visualize rpe during learning

vhats = np.hstack([x['V_hat'].detach().numpy() for x in training_data[0]])
rpes = np.hstack([x['rpe'].detach().numpy() for x in training_data[0]])
X = np.hstack([x['X'].detach().numpy() for x in training_data[0]])
# X = np.reshape(X, (-1, X.shape[-1]))
X = np.reshape(np.transpose(X, (1,0,2)), (-1, X.shape[-1]))
# plt.subplot(2,1,1)
# plt.plot(vhats[:80], '.-')
# plt.plot(vhats[-80:], '.-')
# plt.subplot(2,1,2)
# t = 800

t_pre = 5
t_post = 13
r_time = 5
ntrials = 50

D = np.zeros((ntrials, t_pre+t_post))
trial_starts = np.where(X[:,0] == 1)[0]
for i, t in enumerate(trial_starts[:ntrials]):
    xc = X[(t-t_pre):(t+t_post)]
    xc = xc.sum(axis=1)
    # D[i,:] = xc
    D[i,:] = rpes[(t-t_pre):(t+t_post)]
plt.imshow(D)
plt.xticks(np.arange(0, D.shape[1], 5)-1, labels=np.arange(0, D.shape[1], 5) - t_pre)
plt.xlabel('time rel. to odor')
plt.ylabel('trial index')
plt.colorbar()
plt.show()

plt.plot(D[:,t_pre-1], D[:,t_pre+r_time-1], 'k.-')
plt.plot(D[0,t_pre-1], D[0,t_pre+r_time-1], 'k+')
plt.xlabel('RPE to CS')
plt.ylabel('RPE to US')

#%% plot value during learning

episode_index = 0
# n.b. next time "info['data']"" will just be "info"
V = np.hstack([entry['V_hat'] for entry in training_data[episode_index]])
X = np.vstack([entry['X'] for entry in training_data[episode_index]])
n = len(V)

# find times of odor onsets and get value there
ixO = np.where(X[:,0,0]+X[:,0,1] == 1)[0]
ixO = ixO[ixO < len(X)-1]
ixA = (X[ixO,0,0] == 1)
ixB = (X[ixO,0,1] == 1)
Vc = V[ixO]
Rc = X[ixO+1,0,2]

# plot value and whether or not odor was actually rewarded or not
xs = np.arange(len(Vc))
for r in [0,1]:
    mkr = '.' if r == 1 else 'x'
    msz = 5 if mkr == '.' else 3
    ixa = ixA & (Rc == r)
    ixb = ixB & (Rc == r)
    plt.plot(xs[ixa], Vc[ixa], mkr, color='blue', markersize=msz)
    plt.plot(xs[ixb], Vc[ixb], mkr, color='orange', markersize=msz)
plt.xlabel('trial index')
plt.ylabel('value')
plt.xlim([0, 900]), plt.ylim([0.5, 2.1])

#%% probe model

E.nblocks_per_episode = 100; E.nepisodes = 1
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader, auto_readout_lr=auto_readout_lr)#[1:]

#%% plot value over trials as shown in experiment

plt.figure(figsize=(6,3))
clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, trial in enumerate(responses[:200]):
    v = trial.value[trial.iti]
    rewarded = trial.y.sum() > 0 if not E.is_trial_level else responses[i].y.sum()
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

#%% apply PCA to hidden activity

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

pca = fit_pca(responses)
apply_pca(responses, pca)
plt.plot(100*pca.explained_variance_ratio_, 'o')
plt.xlabel('PC index')
plt.ylabel('% variance explained')

#%% visualize PCs

ix = 0; iy = 1 # PC indices to plot
trials = responses[20:50] # plot some subset of trials

Z = np.vstack([trial.Z_pc for trial in trials]) # hidden activity
X = np.hstack([trial.cue for trial in trials]) # stimuli
B = np.hstack([trial.block_index for trial in trials]) # block index
R = np.hstack([trial.y[0] for trial in trials]) # reward

plt.plot(Z[:,ix], Z[:,iy], '-', color='k', alpha=0.5, zorder=-1, linewidth=1)

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i,(z,x,b,r) in enumerate(zip(Z,X,B,R)):
    rew = r > 0 if not E.is_trial_level else R[i] > 0
    plt.plot(z[ix], z[iy], 'o' if rew else 'x', markersize=5, color=clrs[x], alpha=0.5)

plt.axis('equal')
plt.xlabel('$Z_{}$'.format(ix))
plt.ylabel('$Z_{}$'.format(iy))
