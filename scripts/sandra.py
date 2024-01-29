#%% imports

import os.path
import json
import glob
import numpy as np
import torch
from scipy.io import savemat

from train import make_dataloader, probe_model
from model import ValueRNN
from tasks.starkweather import Starkweather

#%% functions to render trials as .mat

def get_col_names(trial):
    cue_names = ['A', 'B', 'C', 'D']
    assert len(cue_names) == trial.X.shape[-1]-1
    col_names = ['trial_index', 'trial_odor', 'trial_iti', 'trial_isi', 'trial_timestep']
    col_names += ['odor{}_on'.format(cue_names[i]) for i in np.arange(trial.X.shape[-1]-1)] + ['r']
    col_names += ['value']
    col_names += ['Z{}'.format(i) for i in np.arange(trial.Z.shape[1])]
    return col_names

def trial_renderer(trial, col_names):
    # trial_index cue iti isi timestep odorA odorB odorC odorD reward value rpe neurons(x50)
    meta_data = np.tile([trial.index_in_episode, trial.cue, trial.iti, trial.isi], (len(trial), 1))
    data = np.hstack([np.arange(len(trial.value))[:,None], trial.X, trial.value, trial.Z])
    rows = np.hstack([meta_data, data])
    assert rows.shape[1] == len(col_names)
    return rows

def trials_to_mat(trials, outfile):
    all_rows = []
    col_names = get_col_names(trials[0])
    for trial in trials:
        rows = trial_renderer(trial, col_names)
        all_rows.append(rows)
    results = {'col_names': col_names, 'trials': np.vstack(all_rows)}
    savemat(outfile, results, do_compression=True)
    return results

#%% load models

datadir = '/Users/mobeets/code/value-rnn-beliefs/data/models/fulltask'
model_files = glob.glob(os.path.join(datadir, '*.json'))
print('Found {} model files.'.format(len(model_files)))

"""
example args:
    {'run_name': 'fulltask_14496447_11', 'save_dir': 'data', 'n_repeats': 1, 'pretrained_modelfile': None, 'random_seed': None, 'experiment': 'starkweather', 'task_index': 2, 'ncues': 4, 'iti_p': 0.125, 'iti_min': 10, 'p_omission_task_2': 0.1, 'ntrials_per_cue': 2500, 'ntrials_per_episode': 20, 'nblocks': 1000, 'reward_time': 5, 'session_type': 'contingency', 'ntrials': 1000, 'recurrent_cell': 'GRU', 'hidden_size': 50, 'sigma_noise': 0.0, 'gamma': 0.93, 'initialization_gain': 0, 'lr': 0.003, 'batch_size': 12, 'n_epochs': 300, 'n_chances': -1, 'time': '2024-01-05 14:48:14', 'weightsfile': 'data/fulltask_14496447_111_value_starkweather_task2_gru_h50_itimin10_4cues-v0.pth', 'weightsfile_initial': 'data/fulltask_14496447_111_value_starkweather_task2_gru_h50_itimin10_4cues-v0_initial.pth'}
"""

for i, model_file in enumerate(model_files):
    print('Processing model {} of {}...'.format(i+1, len(model_files)))

    # load model file
    args = json.load(open(model_file))

    #  create experiment
    E = Starkweather(ncues=args['ncues'],
		ntrials_per_cue=args['ntrials_per_cue'],
		ntrials_per_episode=args['ntrials_per_cue']*args['ncues'],
		omission_probability=args['p_omission_task_2'] if args['task_index'] == 2 else 0.0,
		iti_p=args['iti_p'],
		iti_min=args['iti_min'],
		omission_trials_have_duration=True,
		half_reward_times=False)
    
    # create model
    model = ValueRNN(input_size=E.ncues + E.nrewards,
		output_size=E.nrewards,
		hidden_size=args['hidden_size'], gamma=args['gamma'])
    
    for wnm in ['', '_initial']: # for model weights both after and before training:
        # load model weights
        weights_file = os.path.join(datadir, os.path.split(args['weightsfile' + wnm])[-1])
        state_dict = torch.load(weights_file)
        state_dict['bias'] = torch.tensor([state_dict['bias'].data])
        model.restore_weights(state_dict)
        
        # run the experiment through the model
        dataloader = make_dataloader(E, batch_size=12)
        trials = probe_model(model, dataloader)[1:] # ignore first trial

        # export trial data as .mat
        matfile = weights_file.replace('.pth', '.mat')
        trials_to_mat(trials, matfile)
