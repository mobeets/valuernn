#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:40:43 2022

@author: mobeets
"""
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

from tasks.contingency import Contingency
from model import ValueRNN
from train import make_dataloader, train_model, probe_model
# from plotting import plot_trials, plot_loss, plot_predictions
# from analysis import get_conditions_contingency, get_exemplars

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

import matplotlib as mpl
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


#%% create experiment

Exps = {}
modes = ['conditioning', 'degradation', 'cue-c']
seeds = [555, 666, 666]

for seed, mode in zip(seeds, modes):
    np.random.seed(seed)
    E = Contingency(mode=mode, ntrials=10000, ntrials_per_episode=20)
    Exps[mode] = E

plt.figure(figsize=(2*len(Exps),4))
for c, (mode, E) in enumerate(Exps.items()):
    plt.subplot(1,len(Exps),c+1)
    # plot_trials([trial.X for trial in E.trials[:30]], new_figure=False)
    plt.title(mode, fontsize=10)
plt.tight_layout()

#%% create model

hidden_size = 50 # number of hidden neurons
gamma = 0.9 # discount rate
model = ValueRNN(input_size=E.ncues + int(E.include_reward),
            hidden_size=hidden_size, gamma=gamma)
model.to(device)
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

modes = ['conditioning', 'degradation', 'cue-c']
mode = modes[2]
useConditionedModel = True

try:
    tmp = results
except:
    results = {}
if useConditionedModel and mode in ['degradation', 'cue-c'] and 'conditioning' in results:
    print("WARNING: Loading condititioning weights prior to training.")
    result = results['conditioning']
    ws = result['weights']
    # ix = np.argmin(result['scores'])
    ix = len(ws)-1
    ws_start = ws[ix]
    model.load_state_dict(ws_start)

if '-frozenrnn' in mode:
    print("WARNING: Freezing RNN weights")
    model.freeze_weights('rnn')
else:
    model.unfreeze_weights('rnn')

lr = 0.003
batch_size = 12
dataloader = make_dataloader(Exps[mode.replace('-frozenrnn', '')], batch_size=batch_size)
scores, _, weights = train_model(model, dataloader, lr=lr, epochs=100)
results[mode] = {'scores': scores, 'weights': weights}
# plot_loss(scores)
# conditioning: 0.005; degradation: ???; cue-c: ???

#%% probe model

modes = ['conditioning', 'degradation', 'cue-c']
model_mode = modes[0]
trial_mode = modes[0]

# load model
result = results[model_mode]
ws = result['weights']
ix = np.argmin(result['scores'])
# ix = len(ws)-1
ws_start = ws[ix]
model.load_state_dict(ws_start)

np.random.seed(777)
E_test = Contingency(mode=trial_mode, ntrials=10000, ntrials_per_episode=20, t_padding=10, jitter=0)
dataloader_test = make_dataloader(E_test, batch_size=batch_size)
responses = probe_model(model, dataloader_test, inactivation_indices=None)

results[model_mode]['responses'] = responses

#%% get exemplars

modes = ['conditioning', 'degradation', 'cue-c']
mode = modes[0]

responses = results[mode]['responses']
conditions = get_conditions_contingency(responses, mode)
exemplars = get_exemplars(responses, conditions, index_in_episode=1)

print('\n'.join([str(x) for x in conditions]))

results[mode]['conditions'] = conditions
results[mode]['exemplars'] = exemplars

#%% visualize value estimates

modes = ['conditioning', 'degradation', 'cue-c']
keys = ['value', 'rpe']
# keys = ['rpe']
doSave = False

nrows = len([x for x in modes if x in results])
ncols = len(keys)
if nrows > 1 and ncols == 1: ncols = len(modes); nrows = len(keys)
separate_trials = False

plt.figure(figsize=(3*ncols,3*nrows)); c = 1
for mode in modes:
    if mode not in results:
        continue
    conditions = results[mode]['conditions']
    exemplars = results[mode]['exemplars']

    for key in keys:
        plt.subplot(nrows,ncols,c); c += 1
        plot_predictions(exemplars, conditions, key, separate_trials=separate_trials,
                        show_events=False, yticks=True, show=False, new_figure=False,
                        linestyle='-', alpha=1.0)
        if not separate_trials:
            plt.ylim([-0.4, 0.8])
        plt.title(mode)
        if c % 2 == 1:
            plt.legend(fontsize=6)
plt.tight_layout()
if doSave:
    plt.savefig('plots/selina/value-and-rpes.pdf')

#%% save

import pickle
# pickle.dump(results, open('data/contingency_h50.pickle', 'wb'))

#%% plot heatmap

sortThisData = True
cue = 0
iti = 5
isi = 9
tPre = 2 # number of time steps shown before odor
tPost = 10 # number of time steps shown after reward

trials = [x for x in responses if x.cue==cue and x.iti==iti and x.isi==isi]# and x.y.sum() == 0]
assert trials[0].X[:,:-1].sum() > 0, "cannot use trial without cue"
X_hats = np.dstack([trial.Z[(iti-tPre):(iti+isi+tPost),:] for trial in trials])

# add noise, split into train/test, and then average separately (to cross-validate)
eps = np.random.randn(*X_hats.shape)
X_hats += 0.1*np.std(np.mean(X_hats, axis=2), axis=0)[None,:,None]*eps
ixTrain = np.argsort(np.random.rand(X_hats.shape[-1])) < 0.5*X_hats.shape[-1]
Ztr = X_hats[:,:,ixTrain].mean(axis=-1).T
Zte = X_hats[:,:,~ixTrain].mean(axis=-1).T

Zd = []
tmax = []
for i,(ztr,zte) in enumerate(zip(Ztr, Zte)):
    tmax.append(np.argmax(ztr))
    zc = (zte-zte.min())/(zte.max()-zte.min()) # only affects visualization
    Zd.append(zc)
if sortThisData:
    ixs = np.argsort(tmax)[::-1]
else:
    print("Using indices from previously sorted data")
Zd = np.vstack(Zd)[ixs]

plt.figure(figsize=(3,3))
plt.imshow(Zd, aspect='auto')
plt.xticks(ticks=[tPre,tPre+isi], labels=['Odor', 'Reward'], fontsize=6)
plt.yticks(ticks=[], labels=[])
plt.xlabel('Time rel. to odor', fontsize=6)
plt.ylabel('Neurons', fontsize=6)

#%% plot hidden activity

from plotting import plot_hidden_activity, plot_hidden_activity_3d, plot_pca
from analysis import fit_pca, apply_pca

pca, f, _ = fit_pca(responses)
vis_data = apply_pca(responses, pca, f)
plot_hidden_activity(exemplars[:], conditions, key='Z_pc')#, xind=2)

#%% get all existing model weight files

import glob
modeldir = '/Users/mobeets/Downloads/paper_weights/'
modelfiles = glob.glob(os.path.join(modeldir, '50_*.pt'))
weightfiles = {}
for modelfile in modelfiles:
    pieces = os.path.splitext(os.path.split(modelfile)[1])[0].split('_')
    model_type = '_'.join(pieces[2:])
    if model_type not in weightfiles:
        weightfiles[model_type] = []
    hidden_size = int(pieces[0])
    seed = int(pieces[1])
    weightfiles[model_type].append({'filepath': modelfile, 'hidden_size': hidden_size, 'seed': seed})

#%% predict lick rates per trial

trial_modes = ['conditioning', 'degradation', 'cue-c']
model_types = ['conditioning_weights', 'cue-c_weights', 'degradation_weights']
pre_iti = 5
max_isi = 7
post_reward = 7
cue = 0

lick_rates = {}
for model_type, infiles in weightfiles.items():
    if model_type not in model_types:
        continue
    if model_type == 'initial_weights':
        model_type += '_conditioning'
    
    # create experiment
    trial_mode = next(x for x in trial_modes if x in model_type)
    np.random.seed(666)
    E_test = Contingency(mode=trial_mode, ntrials=1000, iti_min=20, ntrials_per_episode=20, jitter=1, t_padding=0, rew_times=[8]*3)
    dataloader_test = make_dataloader(E_test, batch_size=100)

    if trial_mode not in lick_rates:
        lick_rates[trial_mode] = []
    c_lick_rates = []
    for item in infiles:
        # load model and eval on experiment
        weights = torch.load(item['filepath'], map_location=torch.device('cpu'))
        input_size = weights['rnn.weight_ih_l0'].shape[-1]
        hidden_size = weights['value.weight'].shape[-1]
        weights['bias'] = weights['bias'][None]
        model = ValueRNN(input_size=input_size, hidden_size=hidden_size, gamma=0.83)
        model.restore_weights(weights)
        trials = probe_model(model, dataloader_test, inactivation_indices=None)
        trials = [trial for trial in trials if trial.index_in_episode > 0]

        # create lick rate for each trial
        for i in range(len(trials)-1):
            trial = trials[i]
            if trial.isi < max_isi: # only take longest isi
                continue
            if trial.cue != cue:
                continue
            if trial.y.sum() > 0:
                continue

            crpes = trial.rpe[(trial.iti-pre_iti):(trial.iti+max_isi-1)]
            cvalues = trial.value[(trial.iti-pre_iti):(trial.iti+max_isi-1),0]
            next_values = trials[i+1].value[:post_reward,0]

            lick_rate = np.hstack([cvalues, next_values])
            iti_lick_rate_max = lick_rate[pre_iti-1]
            isi_lick_onset = pre_iti#+ np.abs(np.random.randn()).astype(int)
            
            DA_at_CS = crpes[pre_iti-1]
            lick_rate_mod = (1 + 10*DA_at_CS)
            lick_rate[isi_lick_onset:] = lick_rate[isi_lick_onset:] * lick_rate_mod

            c_lick_rates.append(lick_rate)
    lick_rates[trial_mode].append(c_lick_rates)

#%% visualize lick rate prediction

clrs = {'conditioning': '#00a275', 'degradation': '#ff7834', 'cue-c': '#7771b8'}
names = {'conditioning': 'Conditioning', 'degradation': 'Degradation', 'cue-c': 'Cued Reward'}

plt.figure(figsize=(3.5,2.5))
for trial_mode in clrs.keys():
    c_lick_rates = lick_rates[trial_mode]
    lrs = np.hstack(c_lick_rates)
    mu = np.nanmean(lrs, axis=0)
    se = np.nanstd(lrs, axis=0) / np.sqrt(lrs.shape[0])
    xs = (np.arange(len(mu)) - pre_iti + 1) * 0.5 # convert to seconds
    plt.plot(xs, mu, label=names[trial_mode], color=clrs[trial_mode])
plt.legend(fontsize=8)
plt.xlabel('Time from odor (s)')
plt.ylabel('Predicted lick rate (a.u.)')

#%% load all modelfiles (used in paper)

results = {}
aresults = {}
aresults_value = {}
trial_modes = ['conditioning', 'degradation', 'cue-c']
pre_iti = 10
max_isi = 7

for model_type, infiles in weightfiles.items():
    # if model_type not in ['conditioning_weights', 'cue-c_weights', 'degradation_weights']:
    #     continue
    if model_type == 'initial_weights':
        model_type += '_conditioning'
    results[model_type] = []
    aresults[model_type] = []
    aresults_value[model_type] = []
    
    # create experiment
    trial_mode = next(x for x in trial_modes if x in model_type)
    np.random.seed(666)
    E_test = Contingency(mode=trial_mode, ntrials=1000, iti_min=20, ntrials_per_episode=20, jitter=1, t_padding=0, rew_times=[8]*3)
    dataloader_test = make_dataloader(E_test, batch_size=100)

    for item in infiles:
        # load model and eval on experiment
        weights = torch.load(item['filepath'], map_location=torch.device('cpu'))
        input_size = weights['rnn.weight_ih_l0'].shape[-1]
        hidden_size = weights['value.weight'].shape[-1]
        weights['bias'] = weights['bias'][None]
        model = ValueRNN(input_size=input_size, hidden_size=hidden_size, gamma=0.83)
        model.restore_weights(weights)

        # np.random.seed(item['seed'])
        # E_test = Contingency(mode=trial_mode, ntrials=1000, ntrials_per_episode=20, jitter=1, iti_min=20, t_padding=0, rew_times=[8, 8, 8])
        # dataloader_test = make_dataloader(E_test, batch_size=100)

        trials = probe_model(model, dataloader_test, inactivation_indices=None)
        # trials = [trial for trial in trials if trial.index_in_episode > 0]

        # get avg RPE in response at the time of each cue onset
        rpes = {}
        arpes = {}
        avalues = {}
        for trial in trials:
            if trial.cue not in rpes:
                rpes[trial.cue] = []
                arpes[trial.cue] = []
                avalues[trial.cue] = []
            if trial.isi < max_isi: # only take longest isi
                continue

            rpes[trial.cue].append(trial.rpe[trial.iti-1][0])

            crpes = trial.rpe[(trial.iti-pre_iti):(trial.iti+max_isi-1)]
            arpes[trial.cue].append(crpes)
            cvalues = trial.value[(trial.iti-pre_iti):(trial.iti+max_isi-1)]
            avalues[trial.cue].append(cvalues)
        for cue in rpes:
            rpes[cue] = np.hstack(rpes[cue])
            arpes[cue] = np.hstack(arpes[cue])
            avalues[cue] = np.hstack(avalues[cue])
        
        results[model_type].append(rpes)
        aresults[model_type].append(arpes)
        aresults_value[model_type].append(avalues)

#%% plot avg results across models

xkeys = ['conditioning_weights', 'degradation_weights', 'cue-c_weights']#, 'initial_weights_conditioning', 'naive_degradation_weights', 'naive_cue-c_weights']
xlbls = ['Conditioning', 'Degradation', 'Cued Reward', 'Initial', 'Naive Degradation', 'Naive Cued Reward']
clrs = ['#00a275', '#ff7834', '#7771b8', 'black', 'black', 'black']

plt.figure(figsize=(6,4.5), dpi=300)
pts = {}
for model_type, res in results.items():
    trial_mode = model_type
    if trial_mode not in xkeys:
        continue

    # get RPE from each model, grouping by (trial_mode, cue)
    for rpes in res:
        for cue in [0,1]:
            if (trial_mode,cue) not in pts:
                pts[(trial_mode,cue)] = []
            mu = np.mean(rpes[cue])
            pts[(trial_mode,cue)].append(mu)

            plt.subplot(1,2,cue+1)
            x = xkeys.index(trial_mode)
            plt.plot(x + 0.5*np.random.rand() - 0.25, mu, 'k.', alpha=0.2)

# plot avg RPE for each (trial_mode, cue) pair
for (trial_mode,cue) in pts:
    plt.subplot(1,2,cue+1)
    cpts = pts[(trial_mode,cue)]
    mu = np.mean(cpts)
    se = np.std(cpts)# / np.sqrt(len(cpts))
    x = xkeys.index(trial_mode)
    plt.plot(x*np.ones(2), [mu-se, mu+se], 'k-')
    plt.bar(x, mu, color=clrs[x])

plt.subplot(1,2,1)
plt.xticks(ticks=range(len(xkeys)), labels=xlbls[:len(xkeys)], rotation=45, ha='right')
plt.ylabel('RPE')
plt.title('Odor A Response')
plt.ylim([-0.02, 0.2])

plt.subplot(1,2,2)
plt.xticks(ticks=range(len(xkeys)), labels=xlbls[:len(xkeys)], rotation=45, ha='right')
plt.ylabel('RPE')
plt.title('Odor B Response')
plt.ylim([-0.4, 0.2])

plt.tight_layout()

#%% plot predicted lick rates

plt.figure(figsize=(5,3))

showLicks = True
# thresh = 0.01
thresh = 0.01

cresults = aresults
# cresults = aresults_value

cue = 0
keys = ['conditioning_weights', 'degradation_weights', 'cue-c_weights']
clrs = ['#00a275', '#ff7834', '#7771b8']

for j, model_type in enumerate(keys):
    all_rpes = cresults[model_type]
    all_values = aresults_value[model_type]
    alicks = []
    for l, arpes in enumerate(all_rpes):
        crpes = arpes[cue]
        cvalues = all_values[l][cue]

        if showLicks:
            licks = []
            for m in range(crpes.shape[1]):
                rps = crpes[:,m] # rpe on current trial
                vls = cvalues[:,m] # value on current trial
                clicks = np.zeros(len(rps))
                if any(rps > thresh):
                    t = next(i for i in range(len(rps)) if rps[i] > thresh)
                    xs = np.arange(len(clicks)-t)

                    # add jitter to onset
                    # t += np.abs(np.random.randn()).astype(int)
                    clicks[t:] = 1 # np.exp(-0.01*xs)
                    # clicks[t] = 1
                licks.append(clicks)
            licks = np.vstack(licks).T
            # licks = (crpes > thresh)
        else:
            licks = crpes
        licks = licks.mean(axis=1)
        alicks.append(licks)
    
    alicks = np.vstack(alicks)
    licks = alicks.mean(axis=0)
    
    plt.subplot(1,2,1)
    xs = (np.arange(len(licks)) - pre_iti + 2) * 0.5 # convert to seconds
    yjitter = (j-1)/100 # so we can see overlapping traces
    plt.plot(xs, licks + yjitter, color=clrs[j], label=model_type.replace('_weights', ''), zorder=-j)
    if showLicks:
        plt.ylabel('lick rate')
    else:
        plt.ylabel('RPE')

    plt.subplot(1,2,2)
    if showLicks:
        plt.bar(j, licks[(pre_iti-1):].sum(), color=clrs[j])
        plt.ylabel('anticipatory lick rate')
    else:
        plt.bar(j, licks[(pre_iti-1)], color=clrs[j])
        plt.ylabel('RPE at CS')

plt.subplot(1,2,1)
plt.xlabel('time rel. to odor onset')
plt.xlim([-2, max(xs)])
# plt.xlim([-2, 3.5])
plt.legend(fontsize=8)
plt.subplot(1,2,2)
plt.xticks(ticks=range(3), labels=[k.replace('_weights', '') for k in keys], rotation=45, ha='right')
plt.tight_layout()

#%% plot predicted lick rates v0

plt.figure(figsize=(5,3))

showLicks = True
# thresh = 0.01
thresh = 0.01

cue = 0
keys = ['conditioning_weights', 'degradation_weights', 'cue-c_weights']
clrs = ['#00a275', '#ff7834', '#7771b8']

for j, model_type in enumerate(keys):
    all_rpes = aresults[model_type]
    alicks = []
    for arpes in all_rpes:
        crpes = arpes[cue]
        # plt.subplot(3,1,j+1)

        if showLicks:
            licks = []
            for rps in crpes.T:
                clicks = np.zeros(len(rps))
                if any(rps > thresh):
                    t = next(i for i in range(len(rps)) if rps[i] > thresh)
                    xs = np.arange(len(clicks)-t)

                    # add jitter to onset
                    t += np.abs(np.random.randn()).astype(int)
                    clicks[t:] = 1 # np.exp(-0.01*xs)
                    # clicks[t] = 1
                licks.append(clicks)
            licks = np.vstack(licks).T
            # licks = (crpes > thresh)
        else:
            licks = crpes
        licks = licks.mean(axis=1)
        alicks.append(licks)
    
    alicks = np.vstack(alicks)
    licks = alicks.mean(axis=0)
    
    plt.subplot(1,2,1)
    xs = (np.arange(len(licks)) - pre_iti + 2) * 0.5 # convert to seconds
    yjitter = (j-1)/100 # so we can see overlapping traces
    plt.plot(xs, licks + yjitter, color=clrs[j], label=model_type.replace('_weights', ''), zorder=-j)
    if showLicks:
        plt.ylabel('lick rate')
    else:
        plt.ylabel('RPE')

    plt.subplot(1,2,2)
    plt.bar(j, licks[(pre_iti-1):].sum(), color=clrs[j])
    plt.ylabel('anticipatory lick rate')

plt.subplot(1,2,1)
plt.xlabel('time rel. to odor onset')
plt.xlim([-2, max(xs)])
# plt.xlim([-2, 3.5])
plt.legend(fontsize=8)
plt.subplot(1,2,2)
plt.xticks(ticks=range(3), labels=[k.replace('_weights', '') for k in keys], rotation=45, ha='right')
plt.tight_layout()

#%% load all modelfiles (used in paper)

results = {}
trial_modes = ['conditioning', 'degradation', 'cue-c']
for model_type, infiles in weightfiles.items():
    if model_type not in ['conditioning_weights', 'cue-c_weights', 'degradation_weights']:
        continue
    # if model_type == 'initial_weights':
    #     model_type += '_conditioning'
    results[model_type] = []
    
    # create experiment
    trial_mode = next(x for x in trial_modes if x in model_type)
    # np.random.seed(666)
    # E_test = Contingency(mode=trial_mode, ntrials=1000, ntrials_per_episode=20, iti_min=20, jitter=1, t_padding=0, rew_times=[8]*3)
    # dataloader_test = make_dataloader(E_test, batch_size=100)

    for item in infiles:
        # load model and eval on experiment
        weights = torch.load(item['filepath'], map_location=torch.device('cpu'))
        input_size = weights['rnn.weight_ih_l0'].shape[-1]
        hidden_size = weights['value.weight'].shape[-1]
        weights['bias'] = weights['bias'][None]
        model = ValueRNN(input_size=input_size, hidden_size=hidden_size, gamma=0.83)
        model.restore_weights(weights)

        np.random.seed(item['seed'])
        E_test = Contingency(mode=trial_mode, ntrials=1000, iti_min=20, ntrials_per_episode=20, jitter=1, t_padding=0, rew_times=[8, 8, 8])
        dataloader_test = make_dataloader(E_test, batch_size=100)

        trials = probe_model(model, dataloader_test, inactivation_indices=None)
        # trials = [trial for trial in trials if trial.index_in_episode > 0]

        # get avg RPE in response at the time of each cue onset
        rpes = {}
        for trial in trials:
            if trial.cue not in rpes:
                rpes[trial.cue] = []
            if trial.isi != 8:
                continue
            if trial.cue == 0 and trial.y.sum() != 1:
                continue
            if trial.iti < 20:
                continue
            rpes[trial.cue].append(trial.rpe[trial.iti-5:])
            # rpes[trial.cue].append(trial.value[trial.iti-5:])
        for cue in rpes:
            rpes[cue] = np.hstack(rpes[cue])
        
        results[model_type].append(rpes)

#%%

plt.figure(figsize=(5,5))

cue = 0
keys = ['conditioning_weights', 'degradation_weights', 'cue-c_weights']
clrs = ['#00a275', '#ff7834', '#7771b8']

for j, model_type in enumerate(keys):
    all_rpes = results[model_type]
    plt.subplot(2,2,j+1)

    cys = []
    for k, rpes in enumerate(all_rpes):
        # if k != 9:
        #     continue
        plt.plot(rpes[cue], '.-', linewidth=1, alpha=0.5)
        cy = rpes[cue].mean(axis=1)
        plt.plot(cy, '.-', linewidth=1, alpha=0.5)
        # cy = rpes[0][4,:1]
        # plt.plot(j * np.ones(len(cy)), cy, '.')
        cys.append(cy)
        
    cys = np.vstack(cys)
    plt.ylim([-0.5, 1.0])
    plt.title(model_type)

    plt.subplot(2,2,4)
    plt.plot(cys.mean(axis=0), '-', linewidth=2, label=model_type, color=clrs[j], zorder=-j)
    plt.ylim([-0.2, 0.8])

# plt.xticks(ticks=range(len(keys)), labels=keys, rotation=90, ha='right')
plt.tight_layout()

#%%
