#%% imports

import os.path
import pickle
import datetime
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import make_dataloader, train_model, probe_model
from model import ValueRNN
from tasks.example import Example

#%% define functions

def get_experiments_by_id(id, default_exp):
    """
    n.b. experiments are encoded as ((iti,isi), (iti,isi), ...)
    4,5,8,9,11,13
    """
    if id == 0:
        if len(default_exp) == 0:
            raise Exception('experiments must be provided either using --experiments or --tasks_by_id')
        return default_exp
    elif id == 1: # fixed I/T and T
        return ((15,3), (20,4), (30,6), (40,8), (60,12), (24,3), (24,4), (24,6), (24,8), (24,12))
    elif id == 2: # fixed I/T = 5
        return ((15,3), (20,4), (30,6), (40,8), (60,12))
    elif id == 3: # fixed I = 24
        return ((24,3), (24,4), (24,6), (24,8), (24,12))
    elif id == 4: # fixed T = 6
        return ((15,6), (20,6), (30,6), (40,6), (60,6))
    elif id == 5: # fixed T = 6, larger I/T
        return ((72,6), (90,6), (120,6), (150,6), (180,6))
    elif id == 6: # ~ Burke 2023 task with time step = 0.25s
        # n.b. had to change 48 -> 29 so we can have equal session lengths
        return ((29*5,5), (290*5,5))
    elif id == 7: # fixed I/T = 10
        return ((30,3), (40,4), (60,6), (80,8), (120,12))
    elif id == 8: # fixed I == 48
        return ((48,3), (48,4), (48,6), (48,8), (48,12))
    elif id == 9: # fixed T = 12
        return ((15,12), (20,12), (30,12), (40,12), (60,12))
    elif id == 10: # fixed I/T = 10, larger T
        return ((150,15), (180,18), (240,24), (300,30), (400,40))
    elif id == 11: # fixed I == 48, larger T
        return ((48,15), (48,18), (48,24), (48,30), (48,40))
    elif id == 12: # fixed I/T = 5, larger T
        return ((75,15), (90,18), (120,24), (150,30), (200,40))
    elif id == 13: # fixed I = 24, larger T
        return ((24,15), (24,18), (24,24), (24,30), (24,40))
    elif id == 14: # fixed I = 24, all T
        return ((24,4), (24,6), (24,8), (24,16), (24,18), (24,24), (24,32), (24,46), (24,60))
    elif id == 15: # fixed I/T = 10, all T
        return ((30,3), (40,4), (60,6), (80,8), (120,12), (160,16), (180,18), (240,24), (320,32), (480,48), (640,64))
    elif id == 16:
        return ((180, 18), (360, 36), (18, 18), (18, 36))
    else:
        raise Exception('experiments id not recognized')
    pass

def check_for_ways_to_reduce_ep_length(experiments):
    trial_lengths = [isi+iti for isi,iti in experiments]
    ep_lengths = []
    for trial_length in trial_lengths:
        other_trial_lengths = [x for x in trial_lengths if x!=trial_length]
        ep_lengths.append(math.lcm(*other_trial_lengths))
    cur_ep_length = math.lcm(*trial_lengths)
    better_ep_length = min(ep_lengths)
    if better_ep_length < cur_ep_length:
        ind = np.argmin(ep_lengths)
        iti, isi = experiments[ind]
        print(f'Could reduce the episode length from {cur_ep_length=} to {better_ep_length=} by removing/modifying ({iti=}, {isi=})')

def get_fixed_episode_length(experiments, args):
    """
    if using a fixed episode length, we must ensure that all experiments
        will have an integer number of trials, where trial length = isi+iti
    if the provided fixed_episode_length is invalid based on the above,
        we will use the least common multiple of all isi+iti
    """
    fixed_episode_length = args['fixed_episode_length']
    if fixed_episode_length < 0:
        return fixed_episode_length, True
    
    is_int = lambda x: x == int(x)
    trial_lengths = [isi+iti for isi,iti in experiments]
    if all([is_int(fixed_episode_length / trial_length) for trial_length in trial_lengths]):
        # all episodes divide evenly; no problems
        return fixed_episode_length, True

    if not args['enforce_complete_trials']:
        # who cares if episodes can't be evenly divided into complete trials
        return fixed_episode_length, False

    # find a better fixed episode length
    old_fixed_episode_length = fixed_episode_length
    fixed_episode_length = math.lcm(*trial_lengths)
    print(f"WARNING: Using {fixed_episode_length=} instead of {old_fixed_episode_length=} so we can evenly divide {experiments=} into integer numbers of trials.")
    check_for_ways_to_reduce_ep_length(experiments)
    return fixed_episode_length, True

def trim_experiment(E, fixed_episode_length):
    assert len(E.episodes) == 1
    trial_durs = np.array([len(trial) for trial in E.episodes[0]])
    trial_dur_cumsum = np.cumsum(trial_durs)
    assert sum(trial_durs[:-1]) < fixed_episode_length
    if sum(trial_durs) == fixed_episode_length:
        return E, 0
    assert sum(trial_durs) > fixed_episode_length
    n_last = fixed_episode_length - sum(trial_durs[:-1])
    E.episodes[0][-1].X = E.episodes[0][-1].X[:n_last]
    E.episodes[0][-1].y = E.episodes[0][-1].y[:n_last]
    return E, trial_durs[-1] - n_last

def make_trials(args):
    Es = {}
    experiments = get_experiments_by_id(args['tasks_by_id'], args['experiments'])
    fixed_episode_length, divides_evenly = get_fixed_episode_length(experiments, args)
    for iti, isi in experiments:
        print('I={}, T={}, I/T={}'.format(iti, isi, iti/isi))
        key = (iti, isi)
        if fixed_episode_length < 0:
            E = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=args['fixed_ntrials_per_episode'], ntrials_per_episode=args['fixed_ntrials_per_episode'], do_trace_conditioning=False)
        else:
            ntrials = np.ceil(fixed_episode_length / (iti+isi)).astype(int)
            E = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=ntrials, ntrials_per_episode=ntrials, do_trace_conditioning=False)
            if not divides_evenly:
                # trim last trial based on fixed_episode_length
                E, n_trimmed = trim_experiment(E, fixed_episode_length)
                print(f'WARNING: Trimed {n_trimmed} time steps off of last trial of {iti=}, {isi=} experiment')
            print(len(E.episodes[0]))
        Es[key] = E
    return Es

def make_model(Es, args):
    E = list(Es.values())[0]
    model = ValueRNN(input_size=E.ncues + E.nrewards,
                    output_size=E.nrewards,
                    hidden_size=args['hidden_size'], gamma=args['gamma'])
    return model

def train_models(model, Es, args):
    results = []
    model.reset(seed=args['seed'])
    W0 = model.checkpoint_weights()
    for key, E in Es.items():
        if args['seed'] is not None:
            model.restore_weights(W0)
            # could say model.reset(args['seed']), but let's be direct
        else:
            model.reset()
        E.make_trials()
        dataloader = make_dataloader(E, batch_size=1)
        epochs = args['nepochs'] if args['nepochs'] > 0 else int(args['ntraining_trials'] / E.ntrials)
        if args.get('verbose', True):
            print(f'----- {key}, epochs={epochs} -----')
        if args['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        elif args['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scores, other_scores, weights = train_model(model, dataloader, optimizer=optimizer, epochs=epochs, print_every=epochs)

        CS_resps = []
        RPE_resps = []
        for W in weights:
            model.restore_weights(W)
            # probe model to get CS response
            trials = probe_model(model, dataloader) # ignore first trial
            CS_resps_cur = np.array([trial.value[trial.iti] for trial in trials])
            RPE_resps_cur = np.array([trial.rpe[trial.iti-1] for trial in trials])
            CS_resps.append(CS_resps_cur)
            RPE_resps.append(RPE_resps_cur)

        CS_resps = np.hstack(CS_resps).T
        RPE_resps = np.hstack(RPE_resps).T
        result = {'key': key, 'scores': scores.copy(), 'CS_resps': CS_resps, 'RPE_resps': RPE_resps}
        results.append(result)
    return results

def save_results(results, args, outfile=None):
    obj = {'args': args, 'results': results}
    obj['time'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if outfile is None:
        outfile = os.path.join(args['outdir'], 'temporal-scaling_' + args['run_name'] + '_' + obj['time'] + '.pickle')
    pickle.dump(obj, open(outfile, 'wb'))

#%% main

def main(args):
    args = dict(**vars(args)) # convert argparse object to dict
    
    if args['verbose']:
        print('making trials...')
    Es = make_trials(args)
    if args['make_trials_only']:
        print('Quitting.')
        return
    
    if args['verbose']:
        print('creating model...')
    model = make_model(Es, args)
    
    if args['verbose']:
        print('training models...')
    results = train_models(model, Es, args)

    if args['verbose']:
        print('saving results...')
    save_results(results, args)

def parse_tuple(s):
    import ast
    try:
        # Use ast.literal_eval to safely evaluate the string into a list of tuples
        result = ast.literal_eval(s)
        if isinstance(result, tuple):
            return result
        else:
            raise argparse.ArgumentTypeError("input must be a tuple of tuples")
    except:
        raise argparse.ArgumentTypeError("input must be a valid tuple of tuples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--ntraining_trials', type=int,
        default=20000,
        help='number of total trials to train on each experiment')
    parser.add_argument('--nepochs', type=int,
        default=-1,
        help='if positive, number of epochs to train on each experiment (ignoring --ntraining_trials)')
    parser.add_argument('--fixed_ntrials_per_episode', type=int,
        default=10,
        help='number of (identical) trials in an episode (aka session). note that gradient steps are taken at the end of each episode (which for our purposes is the same as an epoch)')
    parser.add_argument('--fixed_episode_length', type=int,
        default=-1,
        help='if positive, ensures all experiments have episodes with the same total duration (ignoring --fixed_ntrials_per_episode); if negative, ensures all experiments have episodes with the same number of reinforced trials')
    parser.add_argument('--hidden_size', type=int,
        default=10,
        help='number of hidden units in GRU')
    parser.add_argument('--gamma', type=float,
        default=0.98,
        help='discount factor')
    parser.add_argument('--lr', type=float,
        default=0.0003,
        help='learning rate for gradient descent')
    parser.add_argument('--outdir', type=str,
        default='data',
        help='name of directory in which to save all results as .pickle')
    parser.add_argument('--optimizer', type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='name of optimizer used to find gradient step')
    parser.add_argument('--verbose', action='store_true',
        default=False,
        help='if True, prints all status updates')
    parser.add_argument('--make_trials_only', action='store_true',
        default=False,
        help='if True, makes all trials and then quits')
    parser.add_argument('--enforce_complete_trials', action='store_true',
        default=False,
        help='if --fixed_episode_length is positive, but fixed_episode_length does not allow for all experiments to be divided into complete trials, this will pick a better fixed_episode_length.')
    parser.add_argument('-t', '--tasks_by_id', type=int,
        default=0,
        help='id of experiments (if 0, we use --experiments)')
    parser.add_argument('--experiments', type=parse_tuple,
        required=False,
        default=(),
        help='tuple of (iti,isi) experiments to train models on; e.g. --experiments "((1,2),(2,4))"')
    parser.add_argument('--seed', type=int,
        help='if provided, models will have the same initial weights across all experiments')
    args = parser.parse_args()
    print(args)
    main(args)
