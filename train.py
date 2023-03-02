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

def make_dataloader(experiment, batch_size):
    return DataLoader(experiment, batch_size=batch_size, collate_fn=pad_collate)

def score_epoch(model, dataloader, loss_fn, V_targets):
    V_hats = []
    with torch.no_grad():
        for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):
            V_batch, _ = model(X)
            V_batch = V_batch.numpy()
            V_hat = []
            for i,l in enumerate(x_lengths):
                V_hat += V_batch[:,i][:l]
            V_hats.extend(V_hat)
    return loss_fn(V_hats, V_targets)

def train_epoch(model, dataloader, loss_fn, optimizer=None,
                handle_padding=True, inactivation_indices=None,
                predict_next_input=False, split_rpes=False):
    " if optimizer is None, no gradient steps are taken "
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    train_loss = 0
    n = 0
    if split_rpes:
        print("WARNING: splitting rpes!")
        assert handle_padding, "must handle padding to split rpes"

    for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):
        if predict_next_input:
            X_next = X[1:,:,:]
                        
        if handle_padding:
            # handle sequences with different lengths
            X = pack_padded_sequence(X, x_lengths, enforce_sorted=False)

        # train TD learning
        if not inactivation_indices:
            V, _ = model(X)
        else:
            V, _ = model.forward_with_lesion(X, inactivation_indices)

        if predict_next_input:
            V_hat = V[:-1,:,:]
            V_target = X_next
        else:
            # Compute prediction error
            V_hat = V[:-1,:,:]
            V_next = V[1:,:,:]
            # V_target = y[:-1,:,:] + model.gamma*V_next.detach()
            V_target = y[1:,:,:] + model.gamma*V_next.detach()

        if handle_padding:
            # do not compute loss on padded values
            loss = 0.0
            for i,l in enumerate(x_lengths):
                # we must stop one short because V_target is one step ahead
                if split_rpes:
                    Vp = V_target[:,i][:(l-1)] + F.relu(V_hat[:,i][:(l-1)] - V_target[:,i][:(l-1)])
                    loss += loss_fn(Vp, V_target[:,i][:(l-1)])
                    Vn = V_target[:,i][:(l-1)] + F.relu(V_target[:,i][:(l-1)] - V_hat[:,i][:(l-1)])
                    loss += loss_fn(Vn, V_target[:,i][:(l-1)])
                else:
                    loss += loss_fn(V_hat[:,i][:(l-1)], V_target[:,i][:(l-1)])
            loss /= sum(x_lengths) # when reduction='sum', this makes loss the mean per time step
        else:
            loss = loss_fn(V_hat, V_target)

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

def get_errors(model, dataloader, td_responses, t):
    from analysis import TrialData
    responses = probe_model(model, dataloader, TrialData)
    if t == 0:
        assert all([x.trial_length == y.trial_length for x,y in zip(responses, td_responses)])
    value_err = np.nanmean([(v1-v2)**2 for m1,m2 in zip(responses, td_responses) for v1,v2 in zip(m1.value, m2.value)])
    rpe_err = np.nanmean([(v1-v2)**2 for m1,m2 in zip(responses, td_responses) for v1,v2 in zip(m1.rpe, m2.rpe)])
    # value_rho = np.corrcoef(np.array([(v1[0],v2) for m1,m2 in zip(responses, td_responses) for v1,v2 in zip(m1.value, m2.value)])[:-1].T)[0,1]
    return (value_err, rpe_err)

def train_model(model, dataloader, lr, nchances=4, epochs=5000, handle_padding=True,
                print_every=1, save_every=10, save_hook=None, td_responses=None,
                test_dataloader=None, predict_next_input=False, inactivation_indices=None,
                null_weight=1.0):
    
    if predict_next_input:
        if null_weight < 1:
            print("Warning: giving less weight ({}) in loss to the last input dim.".format(null_weight))
            loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0]*(model.input_size-1) + [null_weight]))
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    
    scores = np.nan * np.ones((epochs+1,))
    scores[0] = train_epoch(model, dataloader, loss_fn, None,
                        handle_padding, predict_next_input=predict_next_input)
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
                    test_score = train_epoch(model, test_dataloader, loss_fn,
                                             None, handle_padding,
                                             predict_next_input=predict_next_input)
                    output += f', test loss: {test_score:0.4f}'
                    other_score += (test_score,)
                if td_responses is not None:
                    (value_err, rpe_err) = get_errors(model, dataloader, td_responses, t)
                    other_score += (value_err, rpe_err)
                    output += f', value error: {value_err:0.3f}, rpe error: {rpe_err:0.3f}'
                other_scores.append(other_score)
                print(output)
            if t % save_every == 0 and save_hook is not None:
                save_hook(model, scores)
            scores[t+1] = train_epoch(model, dataloader, loss_fn, optimizer,
                              handle_padding,
                              predict_next_input=predict_next_input,
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

def probe_model(model, dataloader, predict_next_input=False, inactivation_indices=None):
    responses = []
    model.prepare_to_gather_activity()
    with torch.no_grad():
      for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):

        X_batch = X.numpy()
        y_batch = y.numpy()
        # if inactivation_indices is None:
        #     V_batch, _ = model(X)
        #     Z_batch = model.features['hidden'][0]
        V_batch, Z_batch = model.forward_with_lesion(X, inactivation_indices)
        if model.recurrent_cell == 'LSTM':
            Z_batch = torch.dstack(Z_batch)
        V_batch = V_batch.numpy()
        Z_batch = Z_batch.detach().numpy()
    
        # for each episode in batch
        for j in range(X_batch.shape[1]):
            # extract each trial in episode
            t1 = 0
            for i in range(len(trial_lengths[j])):
                t2 = t1 + trial_lengths[j][i]
                X = X_batch[t1:t2,j,:]
                Z = Z_batch[t1:t2,j,:]
                y = y_batch[t1:t2,j,:]
                V = V_batch[t1:t2,j,:]
                t1 = t2
                
                if predict_next_input:
                    rpe = None
                else:
                    V_hat = V[:-1,:]
                    V_next = V[1:,:]
                    r = y[1:,:]
                    V_target = r + model.gamma*V_next
                    rpe = V_target - V_hat
                
                trial = episode[j][i]
                trial.Z = Z
                trial.value = V
                trial.rpe = rpe
                responses.append(trial)
    return responses
