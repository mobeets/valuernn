#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:13:11 2022

@author: mobeets
"""
#%%
import os.path
import argparse
from argparse import Namespace
import glob
import json
import torch
import numpy as np
import datetime
from copy import deepcopy
import multiprocessing
from multiprocessing.pool import ThreadPool

from model import ValueRNN
from train import make_dataloader, train_model

from tasks.starkweather import Starkweather
from tasks.babayan import Babayan
from tasks.contingency import Contingency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
#%%

def get_filenames(args, run_index):
    model_name = '{}{}_value_{}_task{}_{}_h{}_itimin{}_{}cues{}'.format(
                        args.run_name,
                        run_index,
                        args.experiment,
                        args.task_index if args.experiment == 'starkweather' else '',
                        args.recurrent_cell.lower(), args.hidden_size,
                        args.iti_min, args.ncues, '')
    model_files = glob.glob(os.path.join(args.save_dir, model_name + '*.pth'))
    if model_files:
        max_version = max([int(x.split('-v')[-1].split('.pth')[0]) for x in model_files if '_initial' not in x and '-v' in x])
        version = max_version + 1
    else:
        version = 0
    model_name += '-v{}'.format(version)

    weightsfile_initial = os.path.join(args.save_dir, model_name + '_initial' + '.pth')
    weightsfile = os.path.join(args.save_dir, model_name + '.pth')
    jsonfile = os.path.join(args.save_dir, model_name + '.json')
    return {'jsonfile': jsonfile, 'weightsfile': weightsfile, 'weightsfile_initial': weightsfile_initial}

def save_model_hook(args, run_index):
    files = get_filenames(args, run_index)
    def save_hook(model, scores):

        if not os.path.exists(files['weightsfile_initial']):
            # save initial model weights
            print("Saving initial weights to {}...".format(files['weightsfile_initial']))
            model.save_weights_to_path(files['weightsfile_initial'], model.initial_weights)

        # save best model weights
        print("Saving best weights to {}...".format(files['weightsfile']))
        model.save_weights_to_path(files['weightsfile'], model.saved_weights)

        # save json
        obj = dict(**vars(args))
        obj['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        obj['weightsfile'] = files['weightsfile']
        obj['weightsfile_initial'] = files['weightsfile_initial']
        obj['scores'] = list(scores) # add in scores to obj
        with open(files['jsonfile'], 'w') as f:
            json.dump(obj, f)
    return save_hook

def main_inner(args, run_index):
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
        E.include_null_input = False
        input_size = E.ncues + int(E.include_reward)
    elif args.experiment == 'babayan':
        E = Babayan(nblocks=2*(args.nblocks,),
            ntrials_per_block=2*(5,),
            reward_sizes_per_block=(1,10),
            reward_times_per_block=2*(args.reward_time,),
            jitter=1,
            iti_p=args.iti_p,
            iti_min=args.iti_min,
            include_unique_rewards=False,
            ntrials_per_episode=args.ntrials_per_episode)
        input_size = 1 + int(E.include_reward)
    elif args.experiment == 'contingency':
        E = Contingency(mode=args.session_type,
            ntrials=args.ntrials,
            jitter=1,
            iti_p=args.iti_p,
            iti_min=args.iti_min,
            ntrials_per_episode=args.ntrials_per_episode)
    dataloader = make_dataloader(E, batch_size=args.batch_size)
    
    # create Value RNN
    model = ValueRNN(input_size=input_size,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        bias=True,
        learn_weights=True,
        recurrent_cell=args.recurrent_cell,
        sigma_noise=args.sigma_noise)
    if args.initialization_gain > 0:
        model.initialize(gain=args.initialization_gain)
    if args.pretrained_modelfile:
        print("Loading model weights for initialization...")
        model.load_weights_from_path(args.pretrained_modelfile)
    model.to(device)
    
    # get filenames
    save_hook = save_model_hook(args, run_index)

    # train and save model
    scores, _, weights = train_model(model, dataloader, lr=args.lr,
                                        epochs=args.n_epochs,
                                        nchances=args.n_chances,
                                        save_hook=save_hook)
    if len(scores) < args.n_epochs:
        print("WARNING: Model did not train for all epochs ({} of {}).".format(len(scores), args.n_epochs))
        return

def call_main_inner(**args):
    run_index = args.pop('run_index')
    if type(args) is dict:
        args = Namespace(**args)
    print('======= RUN {} ========'.format(run_index))
    main_inner(args, run_index)

def main(args):
    # If just one run, call it normally
    if args.n_repeats == 1:
        main_inner(args, 1)
        return

    # Runs repeats in parallel
    CPU_COUNT = multiprocessing.cpu_count()
    print("Found {} cpus".format(CPU_COUNT))
    pool = ThreadPool(CPU_COUNT)

    cargs = dict(**vars(args)) # or else the below line changes cargs
    cargs['n_repeats'] = 1
    for i in range(1,args.n_repeats+1):
        targs = dict(**cargs) # or else all cargs are the same
        targs['run_index'] = i
        pool.apply_async(call_main_inner, kwds=targs)

    pool.close()
    pool.join()

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
    parser.add_argument('--pretrained_modelfile', type=str,
        default=None,
        help='modelfile (.pth) to use to initialize weights of model')
    
    # experiment parameters
    parser.add_argument('--random_seed', type=int,
        default=None,
        help='random seed used for generating training data')
    parser.add_argument('--experiment', type=str,
        default='starkweather',
        choices=['starkweather', 'babayan', 'contingency'],
        help='experimental paradigm')
    parser.add_argument('-t', '--task_index', type=int,
        default=1,
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
        help='number of trials per cue (starkweather only)')
    parser.add_argument('--ntrials_per_episode', type=int,
        default=20, help='number of trials per episode')
    
    # babayan task only
    parser.add_argument('--nblocks', type=int,
        default=1000,
        help='number of blocks (babayan only)')
    parser.add_argument('--reward_time', type=int,
        default=5, help='reward time (babayan only)')

    # contingency task only
    parser.add_argument('--session_type', type=str,
        default='contingency',
        choices=['conditioning', 'degradation', 'cue-c'],
        help='type of session to train on (for contingency task only)')
    parser.add_argument('--ntrials', type=int,
        default=1000,
        help='number of trials (for contingency task only)')
    
    # model parameters
    parser.add_argument('-r', '--recurrent_cell', type=str,
        default='GRU',
        choices=['GRU', 'RNN', 'LSTM'],
        help='the recurrent cell used in the rnn')
    parser.add_argument('-k', '--hidden_size',
        type=int,
        default=100,
        help='number of hidden units in the rnn')
    parser.add_argument('-s', '--sigma_noise',
        type=float,
        default=0.0,
        help='std dev of gaussian noise added to hidden units in the rnn')
    parser.add_argument('-g', '--gamma', type=float,
        default=0.93,
        help='gamma (discount factor)')
    parser.add_argument('--initialization_gain', type=float,
        default=0,
        help='initialization gain for recurrent cell')
    
    # training parameters
    parser.add_argument('-l', '--lr', type=float,
        default=0.003,
        help='learning rate for adam')
    parser.add_argument('-b', '--batch_size', type=int,
        default=12,
        help='batch size when training')
    parser.add_argument('-e', '--n_epochs', type=int,
        default=300,
        help='number of epochs when training')
    parser.add_argument('--n_chances', type=int,
        default=-1,
        help='number of epochs to allow loss to increase before stopping early (negative number means no early stopping)')
    
    args = parser.parse_args()
    main(args)
