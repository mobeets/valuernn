#%% task

import numpy as np
from tasks.eshel import Lowet
E = Lowet(ntrials=10000, ntrials_per_episode=20)

#%% make model

from model import ValueRNN
hidden_size = 50 # number of hidden neurons

# import torch; gamma = torch.Tensor([0.5, 0.75, 0.93]) # discount rate
gamma = 0.93
# gamma = 0

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size, gamma=gamma)
model.to('cpu')
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

lmbda = 0 # TD(λ)
from train import make_dataloader, train_model
dataloader = make_dataloader(E, batch_size=12)
scores, _, weights = train_model(model, dataloader, lr=0.003, epochs=1000, lmbda=lmbda)
model.restore_weights(weights[-1])

#%% plot loss

import matplotlib.pyplot as plt
plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

# model.gamma = model.gamma.numpy()
from train import probe_model
E.ntrials_per_episode = E.ntrials
E.jitter = 0
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)#[1:]

#%% plot value/rpe

for c,trial in enumerate(responses[:50]):
    plt.subplot(3,2,trial.cue+1)
    t = trial.iti-2
    plt.plot(trial.rpe[t:,:], '-')
    plt.ylim([-3, 3])
    plt.xticks([1,9], ['cue', 'reward'])
    plt.xlabel('time')
    plt.ylabel('RPE')

plt.tight_layout()
plt.show()

#%% PCA

selector = lambda trial: trial.Z[(trial.iti+1):(trial.iti+trial.isi-1)]

from sklearn.decomposition import PCA

Z = np.vstack([selector(trial) for trial in responses])
pca = PCA(n_components=Z.shape[1])
pca.fit(Z)
plt.plot(pca.explained_variance_ratio_[:10], '.-'), plt.ylim([0,1])

#%% visualize

selector = lambda trial: trial.Z[(trial.iti+1):(trial.iti+trial.isi-1)]

mus = {}
for cue in range(E.ncues):
    zs = np.dstack([pca.transform(selector(trial)) for trial in responses if trial.cue == cue])
    mu = np.mean(np.mean(zs, axis=0), axis=1)
    plt.plot(mu[0], mu[1], 'o', label=cue, alpha=0.5)
    mus[cue] = mu
plt.legend()
plt.axis('equal')

#%% visualize over time

selector = lambda trial: trial.Z[(trial.iti-2):]

for cue in range(E.ncues):
    zs = np.dstack([pca.transform(selector(trial)) for trial in responses if trial.cue == cue])
    zs = np.mean(zs, axis=2)
    
    plt.plot(zs[:,3], '-', label=cue)

    plt.xticks([1,9], ['cue', 'reward'])
    plt.xlabel('time')
    plt.ylabel('Z')

plt.legend()
plt.tight_layout()
plt.show()
