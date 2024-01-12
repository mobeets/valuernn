#%% imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from train import make_dataloader, train_model, probe_model
from model import ValueRNN

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% make trials

from tasks.inference import ValueInference

E = ValueInference(nblocks_per_episode=4, ntrials_per_block=8,
                is_trial_level=True, ntrials_per_block_jitter=3)

#%% make model

hidden_size = 10 # number of hidden neurons

# import torch; gamma = torch.Tensor([0.9, 0.95]) # discount rate
gamma = 0.93 if not E.is_trial_level else 0

model = ValueRNN(input_size=E.ncues + E.nrewards*int(E.include_reward),
                 output_size=E.nrewards, hidden_size=hidden_size, gamma=gamma)
model.to('cpu')
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

epochs = 200
batch_size = 12
dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, optimizer=None, epochs=epochs)
plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss') # plot loss

#%% probe model

# model.gamma = model.gamma.numpy()
E.make_trials() # create new (test) trials
dataloader = make_dataloader(E, batch_size=12)
trials = probe_model(model, dataloader)

#%%
