#%% imports

import torch
import numpy as np
import matplotlib.pyplot as plt
from train import make_dataloader, train_model, probe_model
from model import ValueRNN
from tasks.example import Example

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

ntrials_per_episode = 10
do_trace_conditioning = False # if False, do trace conditioning

# (I,T)
# model_params = [(10,2), (20,4), (30,6), (40,8), (24,2), (24,4), (24,6), (24,8)]
# model_params = [(15,3), (20,4), (30,6), (45,9), (24,3), (24,4), (24,6), (24,9)]
model_params = [(15,3), (20,4), (30,6), (40,8), (60,12), (24,3), (24,4), (24,6), (24,8), (24,12)]
Es = {}
for iti, isi in model_params:
    key = 'T={}, I={}, I/T={}'.format(isi, iti, iti/isi)
    print(key)
    Es[key] = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode, do_trace_conditioning=do_trace_conditioning)

print(len(Es))

# todo: longer ITI, so that we have ITI/ISI of 1, 3, 5, 10, 20 (currently we just have 1 and 3)
# to compare to Fig 1 of it's the information, let's keep a very large ITI fixed, and vary ISI

# E1 = Example(ncues=1, iti=4, isis=[5], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)
# E2 = Example(ncues=1, iti=14, isis=[5], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)
# E3 = Example(ncues=1, iti=9, isis=[10], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)
# E4 = Example(ncues=1, iti=4, isis=[15], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)

# exp_modes = [1,2]
# iti_bases = [48]
# i_over_t_bases = [5]
# # IoverT = [1, 3, 5, 10, 20]
# IoverT = [5, 8, 10]

# iti_bases = [60, 80]
# i_over_t_bases = [5, 15]

# Es = {}
# for exp_mode in exp_modes:
#     for iti_base in iti_bases:
#         for i_over_t_base in i_over_t_bases:
#             print(f'{exp_mode=}, {iti_base=}, {i_over_t_base=}')
#             isis = [int(iti_base/ratio) for ratio in IoverT]
#             if exp_mode == 1:
#                 # ITI fixed
#                 for isi in isis:
#                     iti = iti_base
#                     key = 'T={}, I={}, I/T={}'.format(isi, iti, int(iti/isi))
#                     print(key)
#                     Es[key] = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)
#             elif exp_mode == 2:
#                 # ITI/T fixed
#                 for isi in isis:
#                     iti = int(i_over_t_base*isi)
#                     key = 'T={}, I={}, I/T={}'.format(isi, iti, int(iti/isi))
#                     print(key)
#                     Es[key] = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=ntrials_per_episode, ntrials_per_episode=ntrials_per_episode)
#             elif exp_mode == 3:
#                 # ITI/T fixed, fixed number of trials per episode
#                 for isi in isis:
#                     iti = int(i_over_t_base*isi)
#                     key = 'T={}, I={}, I/T={}'.format(isi, iti, int(iti/isi))
#                     print(key)
#                     c_ntrials_per_episode = int(216 / (iti+isi))
#                     print(c_ntrials_per_episode)
#                     Es[key] = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=c_ntrials_per_episode, ntrials_per_episode=c_ntrials_per_episode)
#             else:
#                 raise Exception('unrecognized exp_mode')

# 72 * 3 = 216 time steps
# 18 -> 3*18 / 3 = 18
# 36 -> 3*36/ 3 = 

# Es = {'short/mid': E1, 'long/low': E2, 'long/mid': E3, 'long/high': E4}
# Es = {'long/low': E2, 'long/mid': E3, 'long/high': E4}
# Es = {'long/mid': E3, 'long/high': E4}
# Es = {'long/low': E2, 'long/mid': E3, 'long/high': E4, 'short/mid': E1}
# Es = {'long/low': E2b, 'long/mid': E3b, 'long/high': E4b}
# Es = {'short/mid': E1, 'long/low': E2}#, 'long/mid': E3, 'long/high': E4}

TC = {k: E.isis[0]/(1+E.iti+E.isis[0]) for k,E in Es.items()}
nf = lambda E: sum([len(x) for x in E.episodes[0]])
print(TC)
print([nf(E) for k,E in Es.items()]) # number of time steps per episode
print([len(E) for k,E in Es.items()]) # number of episodes
# we want E1 to get more reward-pairings per episode
# but for episode duration to be a similar total length
# and to have a total number of episodes

# # gamma = 0.93
# for k,E in Es.items():
#     # r_scale = gamma ** E.isis[0]
#     for episode in E.episodes:
#         for trial in episode:
#             if doDelayConditioning:
#                 trial.X[trial.iti:,0] = 1
#             # trial.y[trial.y > 0] = 1 / r_scale

#%% get true V(CS)

gamma = 0.98
Vtrues = {}
Dtrues = {}
for key in Es:
    E = Es[key]
    X = np.vstack([trial.X for trial in E.episodes[0]])
    R = np.vstack([trial.y for trial in E.episodes[0]])
    V = 0*R
    for i in np.arange(len(R)-1)[::-1]:
        V[i] = R[i] + gamma*V[i+1]
    inds = np.where(X[:,0])[0]
    Vtrues[key] = V[inds][0][0]

    r_inds = np.where(X[:,1])[0]
    Dtrues[key] = V[r_inds][0][0]

    if 'I/T=5' in key:
        h = plt.plot(V, alpha=0.5, label=key)
        plt.plot(inds, V[inds], 'o', alpha=0.5, color=h[0].get_color())
# plt.xlim([0, 200])
plt.legend(fontsize=8)
print(Vtrues)
print(Dtrues)

#%% make model

E = list(Es.values())[0]
hidden_size = 10 # number of hidden neurons
gamma = 0.98 # discount factor
model = ValueRNN(input_size=E.ncues + E.nrewards,
                 output_size=E.nrewards,
                 hidden_size=hidden_size, gamma=gamma)

#%% train model

nreps = 6
batch_size = 1
# lr = 0.0005; total_ntraining_trials = 5000
# lr = 0.001; total_ntraining_trials = 10000
lr = 0.0003; total_ntraining_trials = 12500

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# nreps = 3; total_ntraining_trials = 3000

# results = {}
for key in Es:
    if key not in results:
        results[key] = []

for n in range(nreps):
    print(f'====== RUN {n} ======')
    model.reset()
    W0 = model.checkpoint_weights()
    for key, E in Es.items():
        # if 'T=12' not in key:
        #     continue
        model.restore_weights(W0)
        E.make_trials()
        dataloader = make_dataloader(E, batch_size=batch_size)
        epochs = int(total_ntraining_trials / E.ntrials)
        print(f'----- {key}, epochs={epochs} -----')
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scores, other_scores, weights = train_model(model, dataloader, optimizer=optimizer, epochs=epochs, print_every=epochs)

        CS_resps = []
        RPE_resps = []
        for W in weights:
            model.restore_weights(W)
            # probe model to get CS response
            trials = probe_model(model, dataloader)[1:]
            CS_resps_cur = np.array([trial.value[trial.iti] for trial in trials])
            RPE_resps_cur = np.array([trial.rpe[trial.iti-1] for trial in trials])
            CS_resps.append(CS_resps_cur)
            RPE_resps.append(RPE_resps_cur)

        CS_resps = np.hstack(CS_resps).T
        CS_resp_avg = CS_resps.mean(axis=1)
        RPE_resps = np.hstack(RPE_resps).T
        RPE_resp_avg = RPE_resps.mean(axis=1)
        results[key].append({'scores': scores.copy(), 'weights': weights, 'CS_resp_avg': CS_resp_avg, 'RPE_resp_avg': RPE_resp_avg})

#%% recreate the key Gallistel & Gibbons plots

thresh = 0.1 # to define time to learning
xnse = 0.1 # noise added for jitter
clrs = {'I/T fixed': 'blue', 'I fixed': 'orange'}

PtsT = {}
PtsIoT = []
for key, items in results.items():
    ys = np.vstack([item['RPE_resp_avg'] for item in items])

    T,I,IoT = key.split(',')
    T = float(T.split('=')[1]) # ISI
    I = float(I.split('=')[1]) # ITI
    IoT = I/T # float(IoT.split('=')[1]) # ITI / ISI
    C = I+T # cycle length

    if IoT == 5:
        grp = 'I/T fixed'
    elif I == 24:
        grp = 'I fixed'
    else:
        continue
    if grp not in PtsT:
        PtsT[grp] = []

    for j in range(ys.shape[0]):
        ts = np.where(ys[j,:] > thresh)[0]
        if len(ts) == 0 or ts[0] == 0:
            continue
        y = ts[0]
        PtsT[grp].append((T, y))
        PtsIoT.append((IoT,y))

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
for grp, pts in PtsT.items():
    pts = np.vstack(pts)
    xsa = np.unique(pts[:,0])
    mus = []
    for x in xsa:
        ys = pts[pts[:,0] == x,1]
        mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
        # mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
        mus.append(mu)

        xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
        plt.scatter(xs, ys, s=3, c=clrs[grp], alpha=0.8)
        plt.plot(x, mu, 'o', color=clrs[grp], markeredgecolor='k')
        plt.plot(x * np.ones(2), [lb, ub], 'k-', zorder=-1)
    plt.plot(xsa, mus, '-', c=clrs[grp], zorder=-1, label=grp)

plt.xlabel('T = Delay of Reinforcement')
plt.ylabel('Reinforcements to Acquisition')
plt.xticks(ticks=xsa, labels=[int(x) for x in xsa])
# plt.yscale('log')
plt.legend()

plt.subplot(1,2,2)
pts = np.vstack(PtsIoT)
xsa = np.unique(pts[:,0])
mus = []
for x in xsa:
    ys = pts[pts[:,0] == x,1]
    mu = np.median(ys); lb = np.percentile(ys, 25); ub = np.percentile(ys, 75)
    # mu = np.mean(ys); se = np.std(ys)/np.sqrt(len(ys)); lb = mu-se; ub = mu+se
    mus.append(mu)

    xs = x * np.ones(len(ys)) + xnse*np.random.randn(len(ys))
    plt.scatter(xs, ys, s=3, c='k', alpha=0.8)
    plt.plot(x, mu, 'o', color='k', markeredgecolor='k')
    plt.plot(x * np.ones(2), [lb, ub], 'k-', zorder=-1)
plt.plot(xsa, mus, '-', c='k', zorder=-1, label=grp)

plt.xlabel('I/T')
plt.ylabel('Reinforcements to Acquisition')
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticks=xsa, labels=[int(x) for x in xsa])
plt.minorticks_off()
yticks = [700,500,300]
plt.yticks(ticks=yticks, labels=yticks)
plt.tight_layout()


#%% visualize individual models

showLoss = False

plt.figure(figsize=(6.5,3))
c = 1
for key, items in results.items():
    if 'I/T=5' not in key:
        continue

    if showLoss:
        ys = np.vstack([item['scores'] for item in items])
    else:
        ys = np.vstack([item['RPE_resp_avg'] for item in items])

    plt.subplot(2,4,c); c+=1
    plt.plot(ys.T)
    plt.title(key, fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if not showLoss:
        plt.plot(plt.xlim(), thresh*np.ones(2), 'k-', zorder=-1, alpha=0.5)
    if c == 2:
        plt.xlabel('# reinforcements', fontsize=8)
        plt.ylabel('loss' if showLoss else 'RPE', fontsize=8)

plt.tight_layout()

#%% plot results

plotEachModelTrace = False
ncols = 2; nrows = 3
plt.figure(figsize=(3*ncols,3*nrows))

for doNormToMinMax in [False, True]:
    plt.subplot(nrows,ncols,1+int(doNormToMinMax))
    if doNormToMinMax:
        # fnorm = lambda scs: (scs-scs.min()) / (scs.max() - scs.min())
        fnorm = lambda scs: scs / scs.max()
    else:
        fnorm = lambda scs: scs
    for key, items in results.items():
        scs = np.vstack([fnorm(np.array(item['scores'])) for item in items])
        mus = scs.mean(axis=0)
        ses = scs.std(axis=0) / np.sqrt(scs.shape[0])
        xs = np.arange(len(mus))
        xs *= Es[key].ntrials
        h = plt.plot(xs, mus, label='{}'.format(key))
        if plotEachModelTrace:
            for sc in scs:
                plt.plot(xs, sc, color=h[0].get_color(), alpha=0.4, zorder=-1)
        else:
            plt.fill_between(xs, mus-ses, mus+ses, color=h[0].get_color(), alpha=0.3)
    plt.xlabel('# trials')
    if doNormToMinMax:
        plt.ylabel('loss (normalized)')
    else:
        plt.ylabel('loss')
        plt.legend()

for doScaleToTrue in [False, True]:
    plt.subplot(nrows,ncols,3+int(doScaleToTrue))
    for key, items in results.items():
        ys = np.vstack([item['CS_resp_avg'] for item in items])
        if doScaleToTrue:
            ys /= Vtrues[key]
        mus = ys.mean(axis=0)
        ses = ys.std(axis=0) / np.sqrt(ys.shape[0])
        xs = np.arange(len(mus))
        xs *= Es[key].ntrials
        h = plt.plot(xs, mus, label='{}'.format(key))
        if plotEachModelTrace:
            for y in ys:
                plt.plot(xs, y, color=h[0].get_color(), alpha=0.4, zorder=-1)
        else:
            plt.fill_between(xs, mus-ses, mus+ses, color=h[0].get_color(), alpha=0.3)
    plt.xlabel('# trials')
    if doScaleToTrue:
        plt.ylabel('CS response (norm.)')
        plt.ylim([-0.3, 1.3])
    else:
        plt.ylabel('CS response')
    # plt.xlim([0, 150])
    # plt.ylim([0, 0.1])
    # plt.legend()

for doScaleToTrue in [False, True]:
    plt.subplot(nrows,ncols,5+int(doScaleToTrue))
    for key, items in results.items():
        ys = np.vstack([item['RPE_resp_avg'] for item in items])
        if doScaleToTrue:
            ys /= Dtrues[key]
        mus = ys.mean(axis=0)
        ses = ys.std(axis=0) / np.sqrt(ys.shape[0])
        xs = np.arange(len(mus))
        xs *= Es[key].ntrials
        h = plt.plot(xs, mus, label='{}'.format(key))
        if plotEachModelTrace:
            for y in ys:
                plt.plot(xs, y, color=h[0].get_color(), alpha=0.4, zorder=-1)
        else:
            plt.fill_between(xs, mus-ses, mus+ses, color=h[0].get_color(), alpha=0.3)
    plt.xlabel('# trials')
    if doScaleToTrue:
        plt.ylabel('RPE (norm.)')
        # plt.ylim([-0.3, 1.3])
    else:
        plt.ylabel('RPE')
    # plt.xlim([0, 150])
    # plt.ylim([0, 0.1])
    # plt.legend()

plt.tight_layout()

#%% compare RPE asymptotes

xgrp = 'I/T'
# xgrp = 'T'
# xgrp = 'I'

cgrp = 'I/T'

grp = 'I/T'
# grp = 'I'
grpval = None # if not None, only shows data with this grp value

yIsLearning = True # if False, plots the max DA per model during training
yInv = False # if True, plots the inverse of time to learning
yNorm = False # if True, normalizes to max DA per model during training

thresh = 0.1 # to define time to learning
xnse = 0.15 # noise added for jitter
xlog = True # if x axis is log scaled
ylog = False # if y axis is log scaled

tss = {}
css = {}
for key, items in results.items():
    ys = np.vstack([item['RPE_resp_avg'] for item in items])

    # only keep models where the RPE increased with learning
    # ixKeep = ys[:,-10:].mean(axis=1) > ys[:,:2].mean(axis=1)
    # print(key, ixKeep)
    # ys = ys[ixKeep,:]

    T,I,IoT = key.split(',')
    T = float(T.split('=')[1]) # ISI
    I = float(I.split('=')[1]) # ITI
    C = I+T
    IoT = float(IoT.split('=')[1]) # ITI / ISI
    if xgrp == 'T/C':
        x = T/(I+T)
    elif xgrp == 'I/T':
        x = IoT
    elif xgrp == 'C/T':
        x = C/T
    elif xgrp == 'T':
        x = T
    elif xgrp == 'C':
        x = C
    elif xgrp == 'I':
        x = I
    else:
        raise Exception('invalid xgrp')
    if cgrp == 'T':
        c = T
    elif cgrp == 'I/T':
        c = IoT
    else:
        raise Exception('invalid cgrp')
    if grp == 'I/T':
        g = IoT
    elif grp == 'I':
        g = I
    else:
        raise Exception('invalid grp')
    if grpval is not None and g != grpval:
        continue
    # print(key)

    if x not in tss:
        tss[x] = []
        css[x] = []

    for j in range(ys.shape[0]):
        t = None
        if yIsLearning:
            if yNorm:
                ts = np.where(ys[j,:] > thresh*ys[j,:].max())[0] # rel to max
            else:
                ts = np.where(ys[j,:] > thresh)[0]
            if len(ts) == 0 or ts[0] == 0:
                continue
            t = ts[0]
            if yInv:
                t = 1/t
        else:
            t = ys.max()
        if t is not None:
            # if key == 'T=16, I=80, I/T=5':
            #     print(t,c)
            tss[x].append(t)
            css[x].append(c)

vs_clrs = np.array([c for cs in css.values() for c in cs])
for x, ts in tss.items():
    if len(ts) == 0:
        continue
    mu = np.mean(ts)
    se = np.std(ts) / np.sqrt(len(ts))
    lb = mu-se; ub = mu+se

    mu = np.median(ts)
    lb = np.percentile(ts, 25)
    ub = np.percentile(ts, 75)

    cs = np.array(css[x]).astype(float)
    # print(x, mu, ts)

    xs = x * np.ones(len(ts)) + xnse*np.random.randn(len(ts))
    plt.scatter(xs, ts, s=3, c=cs, alpha=0.8, vmin=vs_clrs.min(), vmax=vs_clrs.max())
    # plt.plot(xs, ts, 'k.', markersize=1, alpha=0.5)
    
    plt.plot(x, mu, 'o', color='gray', markeredgecolor='k')
    plt.plot(x * np.ones(2), [lb, ub], 'k-', zorder=-1)

plt.xlabel(xgrp)
if yIsLearning:
    if yInv:
        plt.ylabel('Inverse of trials acquisition')
    else:
        plt.ylabel('Reinforcement trials to acquisition')
else:
    plt.ylabel('max δ')
if xlog:
    plt.xscale('log')
if ylog:
    plt.yscale('log')
plt.xticks(ticks=list(tss.keys()), labels=list(tss.keys()))
yticks = [100,200,500,1000]
plt.yticks(ticks=yticks, labels=yticks)
plt.minorticks_off()
plt.colorbar()

#%% get CS response during training

ncols = 2; nrows = 1
plt.figure(figsize=(3*ncols,3*nrows)); c = 1

for scaleToVtrue in [False, True]:
    plt.subplot(nrows,ncols,c); c += 1
    for key, items in results.items():
        E = Es[key]
        dataloader = make_dataloader(E, batch_size=1)
        for item in items:
            weights = item['weights']

            CS_resps = []
            for W in weights:
                model.restore_weights(W)
                # probe model to get CS response
                trials = probe_model(model, dataloader)[1:]
                # CS_resps_cur = np.array([trial.rpe[trial.iti-1] for trial in trials])
                CS_resps_cur = np.array([trial.value[trial.iti] for trial in trials])
                CS_resps.append(CS_resps_cur)
            CS_resps = np.hstack(CS_resps).T
            CS_resp_avg = CS_resps.mean(axis=1)

            xs = np.arange(len(CS_resps))
            xs *= Es[key].ntrials

            Vtrue = Vtrues[key]
            scale = Vtrue if scaleToVtrue else 1
            h = plt.plot(xs, CS_resp_avg / scale, label='{}'.format(key))
            if not scaleToVtrue:
                plt.plot(xs, Vtrue + 0*xs, '-', color=h[0].get_color(), zorder=-1, alpha=0.5)

    plt.xlabel('# trials')
    plt.ylabel('CS response' + (' (normalized)' if scaleToVtrue else ''))

plt.legend()
plt.tight_layout()
