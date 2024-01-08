#%% imports

import numpy as np
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
scores, _, weights = train_model(model, dataloader, lr=0.003, epochs=200)

#%% plot loss

plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

# model.gamma = model.gamma.numpy()
from train import probe_model
E.ntrials_per_episode = E.ntrials
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)[1:]

#%% plot value/rpe in Garr2023

rpe = lambda trial: np.abs(np.diff(trial.rpe, axis=1))

plt.figure(figsize=(12,3))
rpes = []
for c,trial in enumerate(responses[:20]):
    if trial.y.sum() == 0:
        continue
    plt.subplot(1,E.ncues+1,trial.cue+1)
    t = trial.iti-2
    # t = 0
    plt.plot(trial.rpe[t:,0], 'r-', label='ND')
    plt.plot(trial.rpe[t:,1], 'b-', label='D')
    plt.plot(rpe(trial)[t:], 'k-', label='|D-ND|')
    rpes.append(rpe(trial)[t:])
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
        plt.title('ITI reward')
plt.subplot(1,E.ncues+1,E.ncues)
for clr,rpe in zip(['orange', 'blue'], rpes):
    plt.plot(rpe, color=clr)
plt.ylim([-1, 1])
plt.xticks([1,10], ['cue' if trial.cue < 2 else '', 'reward'])
plt.xlabel('time')
plt.ylabel('RPE')
plt.tight_layout()

#%% visualize rpes on different trial types

rpe = lambda trial: np.abs(np.diff(trial.rpe[trial.iti-2:,:], axis=1))# - 0.25
for c in range(2):
    rpes = np.hstack([rpe(t) for t in responses if t.cue == c and t.y.sum() == 0])
    plt.plot(rpes.mean(axis=1))

#%% Fig 7B of Garr2023

import scipy.stats
rpe = lambda trial, t: np.abs(np.diff(trial.rpe[t,:]))# - 0.25
# rpe = lambda trial, t: -np.diff(trial.rpe[t,:])
# rpe = lambda trial, t: np.diff(trial.rpe[t,:])**2
# rpe = lambda trial, t: np.sum(trial.rpe[t,:])

plt.figure(figsize=(8,3))
for i in range(4):
    plt.subplot(1,4,i+1)

    if i == 0:
        x1 = [rpe(trial, trial.iti-1) for trial in responses if trial.cue == 0]
        x2 = [rpe(trial, trial.iti-1) for trial in responses if trial.cue == 1]
        nms = ['ND', 'D']
        ttl = 'cue'
    elif i == 1:
        x1 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 0 and trial.y.sum() > 0]
        x2 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 1 and trial.y.sum() > 0]
        nms = ['ND', 'D']
        ttl = 'reward'
    elif i == 2:
        x1 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 0 and trial.y.sum() == 0]
        x2 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 1 and trial.y.sum() == 0]
        nms = ['ND', 'D']
        ttl = 'omission'
    elif i == 3:
        x1 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 2 and trial.y.sum() > 0]
        x2 = [rpe(trial, trial.iti+trial.isi-1) for trial in responses if trial.cue == 1 and trial.y.sum() > 0]
        nms = ['ITI', 'trial']
        ttl = 'ITI vs. trial'
    x1 = np.hstack(x1)
    x2 = np.hstack(x2)

    for j, ys in enumerate([x1,x2]):
        xc = j*np.ones(len(ys))
        xc += 0.1*np.random.randn(len(ys))
        plt.plot(xc, ys, '.', markersize=1, alpha=0.5)
        plt.plot(j + 0.25*np.array([-1,1]), np.median(ys)*np.ones(2), 'k-', linewidth=2)
    
    # wilcoxon rank sum
    _, p = scipy.stats.ranksums(x1, x2)
    if p < 0.0009:
        plt.text(0.5, np.hstack([x1,x2]).max(), '*', fontsize=20)

    plt.xticks(ticks=range(2), labels=nms)
    plt.yticks(ticks=[-1, 0, 1])
    plt.xlim([-0.5, 1.5])
    plt.ylim([-1, 1])
    plt.title(ttl)
    if i == 0:
        plt.ylabel('DA response')
    plt.plot(plt.xlim(), np.zeros(2), 'k-', alpha=0.3, zorder=-1)

plt.tight_layout()

#%% Fig 7C-D of Garr2023

from sklearn import preprocessing
import scipy.linalg

def linreg_fit(X, Y, scale=False, add_bias=True):
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
    if add_bias:
        Z = np.hstack([X, np.ones((X.shape[0],1))])
    else:
        Z = X
    W = scipy.linalg.lstsq(Z, Y)[0]
    return {'W': W, 'scaler': scaler, 'scale': scale, 'add_bias': add_bias}

def rsq(Y, Yhat):
    top = Yhat - Y
    bot = Y - Y.mean(axis=0)[None,:]
    return 1 - np.diag(top.T @ top).sum()/np.diag(bot.T @ bot).sum()

def linreg_eval(X, Y, mdl):
    if mdl['scaler']:
        X = mdl['scaler'].transform(X)
    if mdl['add_bias']:
        Z = np.hstack([X, np.ones((X.shape[0],1))])
    else:
        Z = X
    Yhat = Z @ mdl['W']
    
    # get r-squared
    return {'Yhat': Yhat, 'rsq': rsq(Y, Yhat)}

rpe = lambda trial, t: np.abs(np.diff(trial.rpe[t,:]))# - 0.25 # | y - x |
# rpe = lambda trial, t: np.sum(trial.rpe[t,:])

lags = 2

plt.figure(figsize=(9,3))
for C in range(2):
    X = []
    Y = []
    for t in range(len(responses)-lags):
        trial = responses[t+lags]
        if trial.cue != C:
            continue
        y = rpe(trial, trial.iti+trial.isi-1)
        x = [trial.y.sum() > 0 for trial in responses[t:(t+lags+1)]]
        X.append(x)
        Y.append(y)
    X = np.vstack(X).astype(float)
    Y = np.vstack(Y)

    mdl = linreg_fit(X, Y)
    res = linreg_eval(X, Y, mdl)

    W = mdl['W'][:,0]
    plt.subplot(1,4,1)
    plt.plot(np.arange(-lags, 1), W[:-1])
    plt.plot([-2.5,0.5], np.zeros(2), 'k-', zorder=-1, alpha=0.3)
    plt.xticks(ticks=np.arange(-lags, 1))
    plt.xlabel('Trials back')
    plt.ylabel('Coefficient')

    plt.subplot(1,4,2)
    plt.plot(C, W[-1], 'o')
    plt.xticks(ticks=[0,1], labels=['ND', 'D'])
    plt.xlim([-0.5, 1.5])
    plt.plot([-0.5, 1.5], np.zeros(2), 'k-', zorder=-1, alpha=0.3)
    plt.xlabel('Trial type')
    plt.ylabel('Intercept')

    plt.subplot(1,4,3+C)
    y1 = W[-1] + W[-2]*1 + W[-3]*0
    y2 = W[-1] + W[-2]*1 + W[-3]*1
    y3 = W[-1] + W[-2]*0 + W[-3]*1
    ys = [y1, y2, y3]
    plt.bar(range(len(ys)), ys, color='k')
    plt.xlim([-0.5, 2.5])
    plt.ylim([-0.5, 1])
    plt.plot([-0.5, 2.5], np.zeros(2), 'k-', zorder=-1, alpha=0.3)
    plt.xticks(ticks=[0,1,2], labels=['(1,0)', '(1,1)', '(0,1)'], rotation=90)
    plt.xlabel('Outcome history')
    plt.ylabel('DA response to trial outcome')
    plt.title('nondegraded' if C == 0 else 'degraded')

plt.tight_layout()

# %%
