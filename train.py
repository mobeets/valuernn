#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:43:03 2022

@author: mobeets
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def pad_collate(batch):
    try:
        (xx, yy, trial_lengths, episode) = zip(*batch)
    except:
        (xx, yy, trial_lengths) = zip(*batch)
        episode = None
    x_lengths = [len(x) for x in xx]

    X = pad_sequence(xx, batch_first=True, padding_value=0)
    y = pad_sequence(yy, batch_first=True, padding_value=0)

    X = torch.transpose(X, 1, 0) # n.b. no longer batch_first
    y = torch.transpose(y, 1, 0)
    X = X.float()
    y = y.float()
    return X, y, x_lengths, trial_lengths, episode

def make_dataloader(experiment, batch_size=1):
    return DataLoader(experiment, batch_size=batch_size, collate_fn=pad_collate)

def train_epoch(model, dataloader, loss_fn, optimizer=None, inactivation_indices=None):
    if optimizer is None: # no gradient steps are taken
        model.eval()
    else:
        model.train()

    train_loss = 0
    n = 0
    for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):
        # handle sequences with different lengths
        X = pack_padded_sequence(X, x_lengths, enforce_sorted=False)

        # train TD learning
        V, _ = model(X, inactivation_indices)

        # Compute prediction error
        V_hat = V[:-1,:,:]
        V_next = V[1:,:,:]
        V_target = y[1:,:,:] + model.gamma*V_next.detach()

        # do not compute loss on padded values
        loss = 0.0
        for i,l in enumerate(x_lengths):
            # we must stop one short because V_target is one step ahead
            loss += loss_fn(V_hat[:,i][:(l-1)], V_target[:,i][:(l-1)])
        loss /= sum(x_lengths) # when reduction='sum', this makes loss the mean per time step

        # Backpropagation
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.item()
        
        train_loss += loss
        n += 1
    train_loss /= n
    return train_loss

def train_model(model, dataloader=None,
                experiment=None, batch_size=12, lr=0.003,
                nchances=4, epochs=5000, print_every=1,
                save_hook=None, save_every=10,
                test_dataloader=None, test_experiment=None,
                inactivation_indices=None):
    
    if experiment is not None:
        assert dataloader is None
        dataloader = make_dataloader(experiment, batch_size=batch_size)
    if test_experiment is not None:
        assert test_dataloader is None
        test_dataloader = make_dataloader(test_dataloader, batch_size=batch_size)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    
    scores = np.nan * np.ones((epochs+1,))
    scores[0] = train_epoch(model, dataloader, loss_fn, None)
    best_score = scores[0]
    best_weights = model.checkpoint_weights()
    nsteps_increase = 0
    other_scores = []
    weights = []
    try:
        for t in range(epochs):
            if t % print_every == 0:
                output = f"Epoch {t}, loss: {scores[t]:0.4f}"
                other_score = ()
                if test_dataloader is not None:
                    test_score = train_epoch(model, test_dataloader, loss_fn, optimizer=None)
                    output += f', test loss: {test_score:0.4f}'
                    other_score += (test_score,)
                other_scores.append(other_score)
                print(output)
            if t % save_every == 0 and save_hook is not None:
                save_hook(model, scores)
            scores[t+1] = train_epoch(model, dataloader, loss_fn, optimizer,
                              inactivation_indices=inactivation_indices)
            weights.append(deepcopy(model.state_dict()))
            
            if scores[t+1] < best_score:
                best_score = scores[t+1]
                best_weights = model.checkpoint_weights()
            if scores[t+1] > scores[t]:
                if nsteps_increase > nchances:
                    print("Stopping.")
                    break
                nsteps_increase += 1
            else:
                nsteps_increase = 0
    except KeyboardInterrupt:
        pass
    finally:
        other_scores = np.array(other_scores)
        scores = scores[~np.isnan(scores)]
        model.restore_weights(best_weights)
        save_hook(model, scores)
        print(f"Done! Best loss: {best_score}")
        return scores, other_scores, weights

def probe_model(model, dataloader=None, experiment=None, inactivation_indices=None):
    if experiment is not None:
        assert dataloader is None
        dataloader = make_dataloader(experiment, batch_size=1)
    trials = []
    model.prepare_to_gather_activity()
    with torch.no_grad():
      for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):

        # pass inputs through model while storing hidden activity
        V_batch, _ = model(X, inactivation_indices)
        Z_batch = model.features['hidden'][0]
        if model.recurrent_cell == 'LSTM':
            Z_batch = torch.dstack(Z_batch)
        
        # convert to numpy for saving
        X_batch = X.numpy()
        y_batch = y.numpy()
        V_batch = V_batch.numpy()
        Z_batch = Z_batch.detach().numpy()
    
        # for each episode in batch
        for j in range(X_batch.shape[1]):
            t1 = 0
            # for each trial in episode
            for i in range(len(trial_lengths[j])):
                # get data for current trial
                t2 = t1 + trial_lengths[j][i]
                X = X_batch[t1:t2,j,:]
                Z = Z_batch[t1:t2,j,:]
                y = y_batch[t1:t2,j,:]
                V = V_batch[t1:t2,j,:]
                t1 = t2
                
                # get rpe
                V_hat = V[:-1,:]
                V_next = V[1:,:]
                r = y[1:,:]
                V_target = r + model.gamma*V_next
                rpe = V_target - V_hat
                
                # add data to trial
                trial = episode[j][i]
                trial.Z = Z
                trial.value = V
                trial.rpe = rpe
                trials.append(trial)
    return trials
