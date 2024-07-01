#%% task

import numpy as np
from tasks.eshel import Eshel, DEFAULT_REW_SIZES, DEFAULT_REW_TIMES
from tasks.trial import RewardAmountDistibution, RewardTimingDistribution

REW_SIZES = [1, 2, 3, 5, 10, 20]
REW_TIMES = [8]
rew_size_distributions = [RewardAmountDistibution(REW_SIZES)]
rew_time_distibutions = [RewardTimingDistribution(REW_TIMES)]
E = Eshel(ntrials=10000, ntrials_per_episode=20, cue_shown = [True], cue_probs = [1.0],
          rew_size_distributions=rew_size_distributions, rew_time_distibutions=rew_time_distibutions)

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

from train import make_dataloader, train_model
dataloader = make_dataloader(E, batch_size=12)
scores, _, weights = train_model(model, dataloader, lr=0.003, epochs=1000, alphas=None)#(1.5,0.5))
# model.restore_weights(weights[-1])

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

rsizes = E.rew_size_distributions[0].rew_sizes
clrs = {rsizes[i]: plt.cm.RdBu(i/(len(rsizes)-1)) for i in np.arange(len(rsizes))}
rpes = {r: [] for r in rsizes}

plt.figure(figsize=(3,5))
for c,trial in enumerate(responses[:50]):
    xs = np.arange(len(trial)) - trial.iti
    
    plt.subplot(2,1,1)
    plt.plot(xs, trial.value, '-', color=clrs[trial.reward_size], alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.xlim([-5, 9])

    plt.subplot(2,1,2)
    plt.plot(xs[1:], trial.rpe, '-', color=clrs[trial.reward_size], alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('RPE')
    plt.xlim([-5, 9])
    rpes[trial.reward_size].append(trial.rpe[-1])

plt.tight_layout()
plt.show()

#%% plot avg rpe per reward size

plt.figure(figsize=(3,3))
for r, vs in rpes.items():
    plt.plot(r, np.nanmean(vs), 'o', color=clrs[r], alpha=0.5)
    plt.plot(r*np.ones(len(vs)), vs, '.', color=clrs[r], alpha=0.5)
plt.xlabel('reward size')
plt.ylabel('rpe')
plt.xticks(rsizes)
plt.ylim([-15,15])
