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

def train_model_step_by_step(model, dataloader, epochs=1, optimizer=None, lr=0.003, lmbda=0, inactivation_indices=None, print_every=1, reward_is_offset=True):
    if dataloader.batch_size != 1:
        raise Exception("batch_size must be 1 when training model step-by-step")
    
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
    assert inactivation_indices is None, 'inactivation_indices not implemented'
    episode_losses = []

    model.train()

    try:
        for k in range(epochs):
            # n.b. we don't treat epochs as different from episodes
            for j, (X, y, _, _, _) in enumerate(dataloader):

                h = None
                optimizer.zero_grad() # zero grad between episodes
                cur_episode_losses = []
                for i in range(len(X)-1):

                    # forward pass
                    if model.predict_next_input:
                        # predict next observation
                        vhat, h = model(X[i], h0=h)
                        vtarget = y[i+1 if reward_is_offset else i]
                        h = h.detach()
                    else:
                        # value estimate
                        vhats, hs = model(X[i:(i+2)], h0=h, return_hiddens=True)
                        vhat = vhats[0]
                        vtarget = y[i+1 if reward_is_offset else i] + model.gamma*vhats[1].detach()
                        h = hs[0].detach().unsqueeze(1)
                    loss = loss_fn(vhat, vtarget)
                    
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
                    cur_episode_losses.append(loss.item())

                episode_losses.append(cur_episode_losses)
                losses.append(np.mean(cur_episode_losses))
                if j % print_every == 0:
                    print('Episode {}, loss: {:0.3f}'.format(len(losses), losses[-1]))
    except KeyboardInterrupt:
        pass
    finally:
        return losses, {'episode_losses': episode_losses}, weights

def train_epoch(model, dataloader, loss_fn, optimizer=None, inactivation_indices=None, lmbda=0, reward_is_offset=True):
    if optimizer is None: # no gradient steps are taken
        model.eval()
    else:
        model.train()

    train_loss = 0
    n = 0
    batch_losses = []
    for batch, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):
        # handle sequences with different lengths
        X = pack_padded_sequence(X, x_lengths, enforce_sorted=False)

        # train TD learning
        V, _ = model(X, inactivation_indices)

        if model.predict_next_input:
            # predict next observation
            V_hat = V[:-1,:,:] if reward_is_offset else V
            V_target = y[1:,:,:] if reward_is_offset else y
        else:
            # value estimate
            V_hat = V[:-1,:,:]
            V_next = V[1:,:,:]
            V_target = (y[1:,:,:] if reward_is_offset else y[:-1,:,:]) + model.gamma*V_next.detach()

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
                               reward_is_offset=reward_is_offset)
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
                                                reward_is_offset=reward_is_offset)
                    output += f', test loss: {test_score:0.4f}'
                else:
                    test_score = ()
                test_scores.append(test_score)
                print(output)
            if t % save_every == 0 and save_hook is not None:
                save_hook(model, scores)
            scores[t+1], batch_loss = train_epoch(model, dataloader, loss_fn, optimizer,
                                      inactivation_indices=inactivation_indices,
                                      reward_is_offset=reward_is_offset)
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

def probe_model(model, dataloader=None, experiment=None, inactivation_indices=None, reward_is_offset=True):
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
                trials.append(trial)
    return trials
