#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import traceback

def data_saver(X, y, V_hat, V_target, hs, loss, model, optimizer):
    return {
        'X': X,
        'Z': hs.detach().numpy(),
        'V_hat': V_hat.detach().squeeze(),
        'rpe': V_target.detach().squeeze() - V_hat.detach().squeeze(),
        'loss': loss.item(),
        # 'weights': deepcopy(model.state_dict()),
        # 'optimizer': deepcopy(optimizer.state_dict()['state']),
        }

def data_saver_to_trials(training_trials, training_data, epoch_index=0):
    batch_size = training_data[epoch_index][0]['X'].shape[1]
    if batch_size > 1:
        raise Exception("You must use batch_size==1.")
    X = np.vstack([entry['X'][:,0,:] for entry in training_data[epoch_index]])
    V = np.hstack([entry['V_hat'] for entry in training_data[epoch_index]])
    Z = np.vstack([entry['Z'][:,0,:] for entry in training_data[epoch_index]])
    rpe = np.hstack([entry['rpe'] for entry in training_data[epoch_index]])
    print(X.shape, V.shape, Z.shape, rpe.shape)
    assert len(X) == len(Z)
    assert len(X) == (len(V)+1)

    trials = []
    t_start = 0
    for i, trial in enumerate(training_trials):
        trial = deepcopy(trial)
        t_end = t_start + len(trial)
        Xc = X[t_start:t_end]
        Zc = Z[t_start:t_end]
        Vc = V[t_start:t_end]
        rpec = rpe[t_start:t_end]
        t_pad = len(trial) - len(Xc)
        assert (trial.X[:len(Xc)] == Xc[:len(Xc)]).all()
        if t_pad > 0:
            # some trials will not be processed in full because of stride size
            Xc = np.vstack([Xc, np.nan * np.ones((t_pad, Xc.shape[1]))])
            Zc = np.vstack([Zc, np.nan * np.ones((t_pad, Zc.shape[1]))])
            Vc = np.hstack([Vc, np.nan * np.ones((t_pad,))])
            rpec = np.hstack([rpec, np.nan * np.ones((t_pad,))])
        trial.Z = Zc
        trial.value = Vc
        trial.rpe = rpec
        # trial.readout = W # todo
        trials.append(trial)
        t_start += len(trial)
    return trials

def train_model_TBPTT(model, dataloader, epochs=1, optimizer=None, lr=0.003,
                        lmbda=0, inactivation_indices=None, auto_readout_lr=0.0, print_every=100, reward_is_offset=True, window_size=50, stride_size=1, data_saver=None):
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
    if stride_size < 1:
        raise Exception(f'Provided {stride_size=} is invalid; value must be positive')
    if window_size < 1:
        raise Exception(f'Provided {window_size=} is invalid; value must be positive')
    if stride_size >= window_size:
        raise Exception(f'Provided {stride_size=} is invalid; value must be strictly smaller than {window_size=}')
    
    if model.predict_next_input:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

    losses = []
    losses.append(np.nan)
    weights = []
    weights.append(deepcopy(model.state_dict()))
    data = []

    model.train()
    try:
        for k in range(epochs):
            # n.b. we treat epochs the same as episodes
            # so n losses will be reported where n = epochs*nepisodes
            for j, (X, y, x_lengths, trial_lengths, episode) in enumerate(dataloader):

                finished_episode = False
                h0 = None
                w0 = None
                optimizer.zero_grad() # zero grad between episodes
                cur_window_losses = []
                cur_data = []
                window_ends = range(window_size, len(X)+1, stride_size)
                for c, i in enumerate(window_ends):
                    # get the current window of (X,y), looking back
                    X_window = X[max(0, i-window_size):i]
                    y_window = y[max(0, i-window_size):i]
                
                    # forward pass starting from hidden state at beginning of window
                    h0 = (h0,w0) if auto_readout_lr > 0 else h0
                    V, hs = model(X_window, y=y_window if auto_readout_lr > 0 else None, h0=h0, return_hiddens=True, auto_readout_lr=auto_readout_lr)
                    if auto_readout_lr > 0:
                        hs, ws = hs
                    V_hat = V[:-1]
                    if model.predict_next_input:
                        V_target = (y_window[1:] if reward_is_offset else y_window[:-1])
                    else: # value estimate
                        V_target = (y_window[1:] if reward_is_offset else y_window[:-1]) + model.gamma*V[1:].detach()
                    # hidden state at the beginning of the next window
                    h0 = hs[stride_size-1].detach().unsqueeze(1)
                    if auto_readout_lr > 0:
                        w0 = ws[stride_size-1].detach()#.unsqueeze(1)

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
                    if data_saver is not None:
                        # we want data for each time step, so keep entire first window
                        # also ensure V[t] doesn't depend on gradients at later t
                        is_first_window_in_block = (c == 0)
                        t_start = 0 if is_first_window_in_block else -stride_size
                        cur_data.append(data_saver(X_window[t_start:], y_window[t_start:], V_hat[t_start:], V_target[t_start:], hs[t_start:], loss, model, optimizer))
                    if c % print_every == 0:
                        print('Window {}, loss: {:0.3f}'.format(c, cur_window_losses[-1]))
                
                data.append(cur_data)
                losses.append(np.mean(cur_window_losses))
                weights.append(deepcopy(model.state_dict()))
                finished_episode = True
 
    except KeyboardInterrupt:
        # log this here, so that if we break before an epoch we don't lose everything
        if not finished_episode:
            data.append(cur_data)
            losses.append(np.mean(cur_window_losses))
            weights.append(deepcopy(model.state_dict()))
    except Exception:
        # for any non-keyboard interrupt, we want to see the stack trace
        traceback.print_exc()
    finally:
        return losses, data, weights
