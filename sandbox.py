#%% make trials

from tasks.eshel import Eshel
E = Eshel(ntrials=10000, ntrials_per_episode=20,
          rew_times=[8], cue_shown=[True], cue_probs=[1])

#%% make model

from model import ValueRNN
hidden_size = 5 # number of hidden neurons

import torch
gamma = torch.Tensor([0.9, 0.95]) # discount rate

model = ValueRNN(input_size=E.ncues + int(E.include_reward),
                 output_size=2, hidden_size=hidden_size, gamma=gamma)
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

model.gamma = model.gamma.numpy()
from train import probe_model
E.ntrials_per_episode = E.ntrials
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
responses = probe_model(model, dataloader)[1:]

#%% plot value/rpe

for trial in responses[:1]:
    plt.plot(trial.Z)#[(trial.iti-2):])
    # plt.plot(trial.reward_size, trial.rpe[trial.iti+trial.isi-1:,1], '.')
    # plt.plot(trial.rpe[trial.iti-1,0], trial.rpe[trial.iti-1,1], '.')
    # plt.plot(trial.rpe[trial.iti+trial.isi-1:,0], trial.rpe[trial.iti+trial.isi-1:,1], '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k-')
