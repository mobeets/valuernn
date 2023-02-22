#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:13:11 2022

@author: mobeets
"""
#%%
import os.path
import argparse
import glob
import json
import torch
import numpy as np
from copy import deepcopy

from model import ValueRNN
from train import make_dataloader, train_model

from tasks.starkweather import Starkweather
from tasks.babayan import Babayan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
#%%

def save_model(args, model, scores, weights):
    model_name = '{}_{}_{}_task{}_{}_h{}_itimin{}_{}cues{}'.format(
                        args.run_name, args.rnn_mode, 
                        args.experiment,
                        args.task_index if args.experiment == 'starkweather' else '',
                        args.recurrent_cell.lower(), args.hidden_size,
                        args.iti_min, args.ncues, '_extra' if args.extra_rnn else '')
    model_files = glob.glob(os.path.join(args.save_dir, model_name + '*.pth'))
    if model_files:
        max_version = max([int(x.split('_v')[-1].split('.pth')[0]) for x in model_files if '_initial' not in x])
        version = max_version + 1
    else:
        version = 0
    model_name += '_v{}'.format(version)
    
    # save initial model weights
    outfile = os.path.join(args.save_dir, model_name + '_initial' + '.pth')
    print("Saving initial weights to {}...".format(outfile))
    model.save_weights_to_path(outfile, weights[0])

    # save best model weights
    outfile = os.path.join(args.save_dir, model_name + '.pth')
    print("Saving best weights to {}...".format(outfile))
    model.save_weights_to_path(outfile)

    jsonfile = os.path.join(args.save_dir, model_name + '.json')
    obj = vars(args)
    obj['scores'] = scores # add in scores to obj
    with open(jsonfile, 'w') as f:
        json.dump(obj, f)

def main_inner(args):
    # create experiment
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    if args.experiment == 'starkweather':
        E = Starkweather(ncues=args.ncues,
            ntrials_per_cue=args.ntrials_per_cue,
            ntrials_per_episode=args.ntrials_per_episode,
            omission_probability=args.p_omission_task_2 if args.task_index==2 else 0.0,
            iti_p=args.iti_p,
            iti_min=args.iti_min,
            omission_trials_have_duration=True,
            half_reward_times=False)
        E.include_null_input = args.rnn_mode == 'belief'
        input_size = E.ncues + int(E.include_reward) + int(E.include_null_input)
        output_size = E.ncues + int(E.include_reward) + 1 if E.include_null_input else 1
    elif args.experiment == 'babayan':
        E = Babayan(nblocks=2*(args.ntrials_per_cue,),
            ntrials_per_block=2*(5,),
            reward_sizes_per_block=(1,10),
            reward_times_per_block=2*(5,),
            jitter=1,
            iti_p=args.iti_p,
            iti_min=args.iti_min,
            include_unique_rewards=False,
            ntrials_per_episode=args.ntrials_per_episode)
        input_size = 1 + int(E.include_reward) + int(E.include_unique_rewards)
        output_size = 1
    dataloader = make_dataloader(E, batch_size=args.batch_size)
    
    # create RNN
    model = ValueRNN(input_size=input_size,
                    output_size=output_size,
                    hidden_size=args.hidden_size,
                    gamma=args.gamma,
                    bias=True,
                    learn_weights=True,
                    recurrent_cell=args.recurrent_cell,
                    use_softmax_pre_value=False,
                    sigma_noise=args.sigma_noise,
                    extra_rnn=args.extra_rnn)
    model.to(device)
    
    # train and save model
    scores, _, weights = train_model(model, dataloader, lr=args.lr,
                                        epochs=args.n_epochs,
                                        td_responses=None)
    if len(scores) < args.n_epochs:
        print("Model did not train for all epochs ({} of {}). Quitting without saving.".format(len(scores), args.n_epochs))
        return
    save_model(args, model, scores.tolist(), weights)
    
def main(args):
    print(vars(args))
    for i in range(args.n_repeats):
        print('========RUN {} of {}========'.format(i+1, args.n_repeats))
        main_inner(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('-d', '--save_dir', type=str,
        default='data',
        help='where to save trained model')
    parser.add_argument('-n', '--n_repeats', type=int,
        default=1,
        help='number of models to train')
    
    # experiment parameters
    parser.add_argument('--random_seed', type=int,
        default=None,
        help='random seed used for generating training data')
    parser.add_argument('--experiment', type=str,
        default='starkweather',
        choices=['starkweather', 'babayan'],
        help='experimental paradigm')
    parser.add_argument('-t', '--task_index', type=int,
        default=2,
        choices=[1,2],
        help='task index')
    parser.add_argument('-c', '--ncues', type=int,
        default=1,
        choices=[1,2,3,4],
        help='number of cues')
    parser.add_argument('--iti_p', type=float,
        default=0.125,
        help='iti_p')
    parser.add_argument('--iti_min', type=int,
        default=10,
        help='iti_min')
    parser.add_argument('--p_omission_task_2', type=float,
        default=0.1,
        help='probability of omission when task_index==2')
    parser.add_argument('--ntrials_per_cue', type=int,
        default=10000,
        help='number of trials per cue')
    parser.add_argument('--ntrials_per_episode', type=int,
        default=20,
        help='number of trials per episode')
    
    # model parameters
    parser.add_argument('-m', '--rnn_mode', type=str,
        default='value',
        choices=['value', 'belief'],
        help='which type of rnn to train')
    parser.add_argument('-r', '--recurrent_cell', type=str,
        default='GRU',
        choices=['GRU', 'RNN', 'LSTM'],
        help='the recurrent cell used in the rnn')
    parser.add_argument('-k', '--hidden_size',
        type=int,
        default=100,
        help='number of hidden units in the rnn')
    parser.add_argument('--extra_rnn', action='store_true')
    parser.add_argument('-s', '--sigma_noise',
        type=float,
        default=0.0,
        help='std dev of gaussian noise added to hidden units in the rnn')
    parser.add_argument('-g', '--gamma', type=float,
        default=0.93,
        help='gamma (discount factor)')
    
    # training parameters
    parser.add_argument('-l', '--lr', type=float,
        default=0.003,
        help='learning rate for adam')
    parser.add_argument('-b', '--batch_size', type=int,
        default=12,
        help='batch size when training')
    parser.add_argument('-e', '--n_epochs', type=int,
        default=150,
        help='number of epochs when training')
    
    args = parser.parse_args()
    main(args)
