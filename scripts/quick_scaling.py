#%% imports

import os.path
import pickle
import datetime
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import make_dataloader, train_model, probe_model
from model import ValueRNN
from tasks.example import Example

#%% define functions

def make_trials(args):
    Es = {}
    for iti, isi in args['experiments']:
        print('T={}, I={}, I/T={}'.format(isi, iti, iti/isi))
        key = (isi, iti, iti/isi)
        Es[key] = Example(ncues=1, iti=iti-1, isis=[isi], ntrials=args['ntrials_per_episode'], ntrials_per_episode=args['ntrials_per_episode'], do_trace_conditioning=False)
    return Es

def make_model(Es, args):
    E = list(Es.values())[0]
    model = ValueRNN(input_size=E.ncues + E.nrewards,
                    output_size=E.nrewards,
                    hidden_size=args['hidden_size'], gamma=args['gamma'])
    return model

def train_models(model, Es, args):
    results = []
    model.reset()
    W0 = model.checkpoint_weights()
    for key, E in Es.items():
        model.restore_weights(W0)
        E.make_trials()
        dataloader = make_dataloader(E, batch_size=1)
        epochs = int(args['total_ntraining_trials'] / E.ntrials)
        if args.get('verbose', True):
            print(f'----- {key}, epochs={epochs} -----')
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        scores, other_scores, weights = train_model(model, dataloader, optimizer=optimizer, epochs=epochs, print_every=epochs)

        CS_resps = []
        RPE_resps = []
        for W in weights:
            model.restore_weights(W)
            # probe model to get CS response
            trials = probe_model(model, dataloader)[1:]
            CS_resps_cur = np.array([trial.value[trial.iti] for trial in trials])
            RPE_resps_cur = np.array([trial.rpe[trial.iti-1] for trial in trials])
            CS_resps.append(CS_resps_cur)
            RPE_resps.append(RPE_resps_cur)

        CS_resps = np.hstack(CS_resps).T
        CS_resp_avg = CS_resps.mean(axis=1)
        RPE_resps = np.hstack(RPE_resps).T
        RPE_resp_avg = RPE_resps.mean(axis=1)
        result = {'key': key, 'scores': scores.copy(), 'CS_resp_avg': CS_resp_avg, 'RPE_resp_avg': RPE_resp_avg}
        results.append(result)
    return results

def save_results(results, args, outfile=None):
    obj = {'args': args, 'results': results}
    obj['time'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if outfile is None:
        outfile = os.path.join(args['outdir'], 'temporal-scaling_' + obj['time'] + '.pickle')
    pickle.dump(obj, open(outfile, 'wb'))

#%% main

def main(args):
    args = dict(**vars(args)) # convert argparse object to dict
    
    if args['verbose']:
        print('making trials...')
    Es = make_trials(args)
    
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
    parser.add_argument('--total_ntraining_trials', type=int,
        default=12500,
        help='number of total trials to train each model on')
    parser.add_argument('--ntrials_per_episode', type=int,
        default=10,
        help='n.b. gradient steps are taken at the end of each episode')
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
    parser.add_argument('--verbose', action='store_true',
        default=False,
        help='if verbose, print all status updates')
    parser.add_argument('--experiments', type=parse_tuple,
        required=False,
        default=((15,3), (20,4), (30,6), (40,8), (60,12), (24,3), (24,4), (24,6), (24,8), (24,12)),
        help='tuple of (iti,isi) experiments to train models on; e.g. --experiments "((1,2),(2,4))"')
    args = parser.parse_args()
    print(args)
    main(args)
