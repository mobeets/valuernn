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
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

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

def train_model_TBPTT(model, dataloader, epochs=1, optimizer=None, lr=0.003,
                             lmbda=0, inactivation_indices=None, print_every=100, reward_is_offset=True, window_size=50, stride_size=1, auto_readout_lr=0.0):
    """
    trains RNN using truncated backprop through time
        by providing a window_size (duration over which gradient is computed)
        and stride_size (time steps before we compute the next gradient)
    n.b. if window_size is larger than a given episode,
        we will still take a single gradient step over the entire episode
    n.b. batch_size must be 1 since we assume this is used for online RNN learning
    """
    if dataloader.batch_size != 1:
        raise Exception("batch_size must be 1 when training model with TBPTT")
    if inactivation_indices is not None:
        raise NotImplementedError("inactivation_indices not implemented for TBPTT")
    if auto_readout_lr > 0:
        raise NotImplementedError('auto_readout_lr not implemented for TBPTT')
    if stride_size < 1:
        raise Exception(f'Provided {stride_size=} is invalid; value must be positive')
    if window_size < 1:
        raise Exception(f'Provided {window_size=} is invalid; value must be positive')
    
    if model.predict_next_input:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

    losses = []
    weights = []
    losses.append(np.nan)
    weights.append(deepcopy(model.state_dict()))
    window_losses = []

    model.train()
    try:
        for k in range(epochs):
            # n.b. we treat epochs the same as episodes
            # so n losses will be reported where n = epochs*nepisodes
            for j, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):

                h = None
                optimizer.zero_grad() # zero grad between episodes
                cur_window_losses = []
                for c, i in enumerate(range(0, max(1, len(X)-window_size-1), stride_size)):
                    # get the current sliding window of (X,y)
                    # n.b. we include one time step extra since we trim one off later
                    X_window = X[i:(i+window_size+1)]
                    y_window = y[i:(i+window_size+1)]
                
                    # forward pass
                    V, hs = model(X_window, h0=h, return_hiddens=True)
                    V_hat = V[:-1]
                    if model.predict_next_input:
                        V_target = (y_window[1:] if reward_is_offset else y_window[:-1])
                    else: # value estimate
                        V_target = (y_window[1:] if reward_is_offset else y_window[:-1]) + model.gamma*V[1:].detach()
                    h = hs[stride_size-1].detach().unsqueeze(1)

                    # get loss
                    loss = loss_fn(V_hat, V_target)
                    loss /= len(V_hat) # when reduction='sum', this makes loss the mean per time step
                    
                    # either zero out gradient for TD(0), or decay it for TD(λ)
                    if lmbda == 0: # TD(0)
                        optimizer.zero_grad()
                    else: # TD(λ)
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad *= model.gamma*lmbda

                    # take gradient step
                    loss.backward()
                    optimizer.step()
                    cur_window_losses.append(loss.item())
                
                    if c % print_every == 0:
                        print('Window {}, loss: {:0.3f}'.format(c, cur_window_losses[-1]))
                
                window_losses.append(cur_window_losses)
                losses.append(np.mean(cur_window_losses))
                weights.append(deepcopy(model.state_dict()))
                    
    except KeyboardInterrupt:
        pass
    finally:
        if len(cur_window_losses) > 0:
            window_losses.append(cur_window_losses)
            losses.append(np.mean(cur_window_losses))
            weights.append(deepcopy(model.state_dict()))
        return losses, {'window_losses': window_losses}, weights

def train_epoch(model, dataloader, loss_fn, optimizer=None, inactivation_indices=None, lmbda=0, reward_is_offset=True, auto_readout_lr=0.0, alphas=None):
    if optimizer is None: # no gradient steps are taken
        model.eval()
    else:
        model.train()
    if inactivation_indices is not None and auto_readout_lr > 0:
        raise Exception("Cannot provide inactivation_indices and set auto_readout_lr > 0")

    train_loss = 0
    n = 0
    batch_losses = []
    for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):
        # handle sequences with different lengths
        X = pack_padded_sequence(X, x_lengths, enforce_sorted=False)

        # train TD learning
        V, _ = model(X, inactivation_indices, y=y if auto_readout_lr > 0 else None, auto_readout_lr=auto_readout_lr)

        if model.predict_next_input:
            # predict next observation
            V_hat = V[:-1,:,:] if reward_is_offset else V
            V_target = y[1:,:,:] if reward_is_offset else y
        else:
            # value estimate
            V_hat = V[:-1,:,:]
            V_next = V[1:,:,:]
            V_target = (y[1:,:,:] if reward_is_offset else y[:-1,:,:]) + model.gamma*V_next.detach()

            if alphas is not None:
                # for distributional RL
                alpha_plus, alpha_minus = alphas
                alpha = alpha_plus*(V_target > V_hat) + alpha_minus*(V_target <= V_hat)
                V_hat *= torch.sqrt(alpha)
                V_target *= torch.sqrt(alpha)

        # do not compute loss on padded values
        loss = 0.0
        for i,l in enumerate(x_lengths):
            # we must stop one short because V_target is one step ahead
            loss += loss_fn(V_hat[:,i][:(l-1)], V_target[:,i][:(l-1)])
        loss /= sum(x_lengths) # when reduction='sum', this makes loss the mean per time step

        # Backpropagation
        if optimizer is not None:
            if lmbda == 0:
                # TD(0)
                optimizer.zero_grad()
            else:
                # TD(λ)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad *= model.gamma*lmbda
            loss.backward()
            optimizer.step()

        loss = loss.item()
        
        train_loss += loss
        batch_losses.append(loss)
        n += 1
    train_loss /= n
    return train_loss, batch_losses

def train_model(model, dataloader=None,
                experiment=None, batch_size=12, lr=0.003, lmbda=0,
                nchances=-1, epochs=5000, print_every=1,
                reward_is_offset=True,
                auto_readout_lr=0.0, alphas=None,
                save_hook=None, save_every=10, optimizer=None,
                test_dataloader=None, test_experiment=None,
                inactivation_indices=None):
    
    if experiment is not None:
        assert dataloader is None
        dataloader = make_dataloader(experiment, batch_size=batch_size)
    if test_experiment is not None:
        assert test_dataloader is None
        test_dataloader = make_dataloader(test_dataloader, batch_size=batch_size)

    if model.predict_next_input:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    
    scores = np.nan * np.ones((epochs+1,))
    batch_losses = []
    scores[0], _ = train_epoch(model, dataloader, loss_fn, None, lmbda=lmbda,
                               reward_is_offset=reward_is_offset,
                               auto_readout_lr=auto_readout_lr, alphas=alphas)
    best_score = scores[0]
    best_weights = model.checkpoint_weights()
    nsteps_increase = 0
    test_scores = []
    weights = []
    weights.append(deepcopy(model.state_dict()))
    try:
        for t in range(epochs):
            if t % print_every == 0:
                output = f"Epoch {t}, loss: {scores[t]:0.4f}"
                if test_dataloader is not None:
                    test_score, _ = train_epoch(model, test_dataloader, loss_fn, optimizer=None,
                                                reward_is_offset=reward_is_offset,
                                                auto_readout_lr=auto_readout_lr, alphas=alphas)
                    output += f', test loss: {test_score:0.4f}'
                else:
                    test_score = ()
                test_scores.append(test_score)
                print(output)
            if t % save_every == 0 and save_hook is not None:
                save_hook(model, scores)
            scores[t+1], batch_loss = train_epoch(model, dataloader, loss_fn, optimizer,
                                      inactivation_indices=inactivation_indices,
                                      reward_is_offset=reward_is_offset,
                                      auto_readout_lr=auto_readout_lr, alphas=alphas)
            weights.append(deepcopy(model.state_dict()))
            batch_losses.append(batch_loss)
            
            if scores[t+1] < best_score:
                best_score = scores[t+1]
                best_weights = model.checkpoint_weights()
            if scores[t+1] > scores[t]:
                if nchances > 0 and nsteps_increase > nchances:
                    print("Stopping early.")
                    break
                nsteps_increase += 1
            else:
                nsteps_increase = 0
    except KeyboardInterrupt:
        pass
    finally:
        test_scores = np.array(test_scores)
        scores = scores[~np.isnan(scores)]
        model.restore_weights(best_weights)
        if save_hook is not None:
            save_hook(model, scores)
        print(f"Done! Best loss: {best_score}")
        return scores, {'test_loss': test_scores, 'batch_losses': batch_losses}, weights

def probe_model(model, dataloader=None, experiment=None, inactivation_indices=None, reward_is_offset=True, auto_readout_lr=0.0):
    if experiment is not None:
        assert dataloader is None
        dataloader = make_dataloader(experiment, batch_size=1)
    trials = []
    model.prepare_to_gather_activity()
    with torch.no_grad():
      for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):

        # pass inputs through model while storing hidden activity
        V_batch, Out_batch = model(X, inactivation_indices, return_hiddens=True, y=y if auto_readout_lr > 0 else None, auto_readout_lr=auto_readout_lr)
        Z_batch = model.features['hidden'][0]
        if model.recurrent_cell == 'LSTM':
            Z_batch = torch.dstack(Z_batch)
        if auto_readout_lr > 0:
            W_batch = Out_batch[1].detach()
        else:
            W_batch = torch.tile(model.value.weight.data, (Z_batch.shape[0], Z_batch.shape[1], 1))
        
        # convert to numpy for saving
        X_batch = X.numpy()
        y_batch = y.numpy()
        V_batch = V_batch.numpy()
        Z_batch = Z_batch.detach().numpy()
        W_batch = W_batch.numpy()
    
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
                W = W_batch[t1:t2,j,:]
                t1 = t2
                
                # get rpe
                if model.predict_next_input:
                    rpe = None
                else:
                    V_hat = V[:-1,:]
                    V_next = V[1:,:]
                    r = y[1:,:] if reward_is_offset else y[:-1,:]
                    V_target = r + model.gamma*V_next
                    rpe = V_target - V_hat
                
                # add data to trial
                trial = deepcopy(episode[j][i])
                trial.Z = Z
                trial.value = V
                trial.rpe = rpe
                trial.readout = W
                trials.append(trial)
    return trials
