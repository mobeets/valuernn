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

from tasks.eshel import Eshel
from tasks.trial import RewardAmountDistibution, RewardTimingDistribution
E = Eshel(
        rew_size_distributions=[RewardAmountDistibution([1])]*3,
        rew_time_distibutions=[RewardTimingDistribution([7]), RewardTimingDistribution([9]), RewardTimingDistribution([11])],
        cue_shown=[True]*3,
        cue_probs=np.ones(3)/3,
        jitter=1,
        ntrials=20000,
        ntrials_per_episode=20)

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

epochs = 1
batch_size = 12
from train import make_dataloader, train_model
dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, lr=0.003, epochs=epochs)

#%% plot loss

import matplotlib.pyplot as plt
# plt.plot(scores), plt.xlabel('# epochs')
plt.plot(other_scores['batch_losses'][0]), plt.xlabel('# episodes')
plt.ylabel('loss')

#%% probe model

# model.gamma = model.gamma.numpy()
from train import probe_model
E.ntrials_per_episode = E.ntrials
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)[1:]

#%% collect PSTHs per cue

import numpy as np
from sklearn.decomposition import PCA

Z = np.vstack([trial.Z for trial in responses])
pca = PCA(n_components=Z.shape[1])
pca.fit(Z)
# plt.plot(pca.explained_variance_ratio_[:10], '.-'), plt.ylim([0,1])
n_pcs = 5

Zs = {}
for cue in range(E.ncues):
    V = np.dstack([trial.value[trial.iti:] for trial in responses if trial.cue == cue])
    Z = np.dstack([pca.transform(trial.Z[trial.iti:])[:,:n_pcs] for trial in responses if trial.cue == cue])

    V = V.mean(axis=-1)
    Z = Z.mean(axis=-1)
    Zs[cue] = Z

#%%

import scipy.linalg

maxT = 8
X = np.dstack([Zs[0][:maxT], Zs[2][:maxT]])
X = X.reshape(X.shape[0]*X.shape[1], -1)
X = np.hstack([X, np.ones(len(X))[:,None]])
Y = Zs[1][:maxT].flatten()
print(X.shape, Y.shape)
w = scipy.linalg.lstsq(X, Y)[0]
print(w)
Yhat = X @ w

def rsq(Y, Yhat):
    top = Yhat - Y
    bot = Y - Y.mean(axis=0)[None,:]
    return 1 - np.diag(top.T @ top).sum()/np.diag(bot.T @ bot).sum()

print('R^2 = {:0.3f}'.format(rsq(Y[:,None], Yhat[:,None])))

# %%
