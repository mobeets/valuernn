#%% make trials

from tasks.eshel import Eshel
E = Eshel(ntrials=10000, ntrials_per_episode=20,
          rew_times=[8], cue_shown=[True], cue_probs=[1])

from tasks.contingency_general import ContingencyGeneral
E = ContingencyGeneral(ntrials=10000, ntrials_per_episode=20)

#%% make model

from model import ValueRNN
hidden_size = 5 # number of hidden neurons

import torch
# gamma = torch.Tensor([0.9, 0.95]) # discount rate
gamma = 0.93

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size, gamma=gamma)
model.to('cpu')
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

from train import make_dataloader, train_model
dataloader = make_dataloader(E, batch_size=12)
scores, _, weights = train_model(model, dataloader, lr=0.003, epochs=100)

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

plt.figure(figsize=(8,3))
for c,trial in enumerate(responses[:20]):
    plt.subplot(1,E.ncues,trial.cue+1)
    t = trial.iti-2
    # t = 0
    plt.plot(trial.rpe[t:,0], 'r-', label='ND')
    plt.plot(trial.rpe[t:,1], 'b-', label='D')
    # plt.plot(np.abs(trial.rpe[t:,1] - trial.rpe[t:,0]), 'b-', label='D')
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
        plt.title('No cue')

plt.tight_layout()
plt.show()

# %%
