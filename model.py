#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:43:03 2022

@author: mobeets
"""
from copy import deepcopy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import gru_with_noise

#%%

class ValueRNN(nn.Module):
    def __init__(self, input_size=4, output_size=1, hidden_size=1, 
                 num_layers=1, gamma=0.9, bias=False, learn_weights=False,
                 predict_next_input=False, use_softmax_pre_value=False,
                 recurrent_cell='GRU',
                 sigma_noise=0.0, extra_rnn=False):
      super(ValueRNN, self).__init__()

      self.gamma = gamma
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.recurrent_cell = recurrent_cell
      self.extra_rnn = extra_rnn

      if extra_rnn and recurrent_cell != 'GRU':
          raise Exception("recurrent_cell must be GRU to have an extra RNN")
      elif extra_rnn and sigma_noise > 0:
          raise Exception("sigma_noise must be zero to have an extra RNN")
      if recurrent_cell == 'GRU':
          if sigma_noise > 0:
              self.rnn = gru_with_noise.GRUWithNoise(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sigma_noise=sigma_noise)
          else:
              self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
          if extra_rnn:
              self.rnn_extra = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
      else:
          if sigma_noise > 0:
              raise Exception("recurrent_cell must be GRU to have sigma_noise > 0")
          if recurrent_cell == 'RNN':
              self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
          elif recurrent_cell == 'LSTM':
              self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
          else:
              raise Exception("recurrent_cell options: GRU, RNN, LSTM")

      if learn_weights:
          self.value = nn.Linear(in_features=hidden_size + hidden_size*extra_rnn,
                             out_features=output_size, bias=False)
      else:
          self.value = lambda x: torch.sum(x,2)[:,:,None]
      self.predict_next_input = predict_next_input
      if self.predict_next_input and output_size != input_size:
          raise Exception("output_size must match input_size when predict_next_input == True")
      if self.predict_next_input and not learn_weights:
          raise Exception("learn_weights must be True when predict_next_input == True")
      self.use_softmax_pre_value = use_softmax_pre_value
      self.bias = nn.Parameter(torch.tensor([0.0]*output_size))
      self.bias.requires_grad = bias
      self.saved_weights = {}
      self.reset()

    def forward(self, xin):
        x, hidden = self.rnn(xin)
        if self.extra_rnn:
            x2, hidden2 = self.rnn_extra(xin)
        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x, output_lengths = pad_packed_sequence(x, batch_first=False)
            if self.extra_rnn:
                x2, output_lengths = pad_packed_sequence(x2, batch_first=False)
        if self.extra_rnn:
            x = torch.dstack([x, x2])
    
        if self.predict_next_input:
            # x = F.softmax(x, dim=-1)
            x = self.bias + self.value(x)
            # n.b. cross-entropy applies softmax, so we don't want that here
            return x, hidden
        else:
            if self.use_softmax_pre_value:
                x = F.softmax(x, dim=-1)
            return self.bias + self.value(x), hidden
    
    def forward_with_lesion_inner(self, x, indices, first_rnn=True):
        hs = []
        cs = []
        os = []
        h_t = torch.zeros((1, x.shape[1], self.hidden_size))
        if self.recurrent_cell == 'LSTM':
            c_t = torch.zeros((1, x.shape[1], self.hidden_size))
            send_cell = True
        else:
            send_cell = False
        for x_t in x:
            h_t_cur = (h_t, c_t) if send_cell else h_t
            if first_rnn:
                o_t, h_t = self.rnn(x_t[None,:], h_t_cur)
            else:
                assert(self.extra_rnn)
                o_t, h_t = self.rnn_extra(x_t[None,:], h_t_cur)
            if send_cell:
                (h_t, c_t) = h_t
            if len(h_t) == 0:
                assert self.recurrent_cell == 'GRU' and self.rnn.sigma_noise > 0
                h_t = o_t

            if indices is not None:
                for index in indices:
                    if index > h_t.shape[-1]:
                        raise Exception("Cannot lesions cell units in LSTM")
                    h_t.data[:,:,index] = 0
                    # n.b. can't lesion c_t in isolation b/c h_t depends on c_t
            hs.append(h_t)
            os.append(o_t)
            if send_cell:
                cs.append(c_t)                
            
        hs = torch.vstack(hs)
        os = torch.vstack(os)
        if send_cell:
            cs = torch.vstack(cs)
            hs = (hs, cs)
        return hs, os
    
    def forward_with_lesion(self, x, indices):
        inds = None if indices is None else [x for x in indices if x < self.hidden_size]
        hs, os = self.forward_with_lesion_inner(x, inds)
        if self.extra_rnn:
            inds2 = None if indices is None else [x-self.hidden_size for x in indices if x-self.hidden_size>=0]
            hs2, os2 = self.forward_with_lesion_inner(x, inds2)
            hs = torch.dstack([hs, hs2])
            os = torch.dstack([os, os2])
            
        if self.predict_next_input:
            return self.bias + self.value(os), hs
        else:
            if self.use_softmax_pre_value:
                os = F.softmax(os, dim=-1)
            return self.bias + self.value(os), hs
    
    def forward_with_lesion_old(self, x, indices):
        # note: valid for GRU only!
        # assert self.recurrent_type == "GRU", "Inactivating units requires a GRU!"
        hs = []
        cs = []
        os = []
        h_t = torch.zeros((1, x.shape[1], self.hidden_size))
        if self.recurrent_cell == 'LSTM':
            c_t = torch.zeros((1, x.shape[1], self.hidden_size))
            send_cell = True
        else:
            send_cell = False
        for x_t in x:
            h_t_cur = (h_t, c_t) if send_cell else h_t
            o_t, h_t = self.rnn(x_t[None,:], h_t_cur)
            if send_cell:
                (h_t, c_t) = h_t
            if len(h_t) == 0:
                assert self.recurrent_cell == 'GRU' and self.rnn.sigma_noise > 0
                h_t = o_t

            if indices is not None:
                for index in indices:
                    if index > h_t.shape[-1]:
                        raise Exception("Cannot lesions cell units in LSTM")
                    h_t.data[:,:,index] = 0
                    # n.b. can't lesion c_t in isolation b/c h_t depends on c_t
            hs.append(h_t)
            os.append(o_t)
            if send_cell:
                cs.append(c_t)                
            
        hs = torch.vstack(hs)
        os = torch.vstack(os)
        if send_cell:
            cs = torch.vstack(cs)
            hs = (hs, cs)
        
        if self.predict_next_input:
            return self.bias + self.value(os), hs
        else:
            if self.use_softmax_pre_value:
                os = F.softmax(os, dim=-1)
            return self.bias + self.value(os), hs

    def freeze_weights(self, substr=None):
        for name, p in self.named_parameters():
            if substr is None or substr in name:
                p.requires_grad = False
    
    def unfreeze_weights(self, substr=None):
        for name, p in self.named_parameters():
            if substr is None or substr in name:
                p.requires_grad = True

    def initialize(self, gain=1):
        """
        https://github.com/rodrigorivera/mds20_replearning/blob/0426340725fd55a616b0d40356ddcebe06ed0f24/skip_thought_vectors/encoder.py
        https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/2
        https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5
        https://pytorch.org/docs/stable/nn.init.html
        """
        print("WARNING: Using tensorflow-style initialization of GRU/RNN")
        assert self.num_layers == 1
        assert self.recurrent_cell.lower() in ['gru', 'rnn']
        assert self.kernel_initializer == 'glorot_uniform'
        assert self.recurrent_initializer == 'orthogonal'
        assert self.bias_regularizer == 'zeros'
        for weight_ih, weight_hh, bias_ih, bias_hh in self.rnn.all_weights:
            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)
            for i in range(0, weight_hh.size(0), self.hidden_size):
                nn.init.orthogonal_(weight_hh.data[i:(i+self.hidden_size)], gain=gain) # orthogonal
                nonlinearity = 'tanh' if ((self.recurrent_cell.lower() == 'rnn') or (i == 2)) else 'sigmoid'
                nn.init.xavier_uniform_(weight_ih.data[i:(i+self.hidden_size)], gain=nn.init.calculate_gain(nonlinearity)) # glorot_uniform

    def reset(self):
        self.bias = nn.Parameter(torch.tensor(0.0))
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        self.initial_weights = self.checkpoint_weights()
               
    def checkpoint_weights(self):
        # self.saved_weights = deepcopy(self.state_dict())
        self.saved_weights = pickle.loads(pickle.dumps(self.state_dict()))
        return self.saved_weights
        
    def restore_weights(self, weights=None):
        weights = self.saved_weights if weights is None else weights
        if weights:
            self.load_state_dict(weights)

    def n_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def save_weights_to_path(self, path, weights=None):
        torch.save(self.state_dict() if weights is None else weights, path)
        
    def load_weights_from_path(self, path):
        self.load_state_dict(torch.load(path))
        
    def get_features(self, name):
        def hook(mdl, input, output):
            self.features[name] = output
        return hook
    
    def prepare_to_gather_activity(self):
        if hasattr(self, 'handle'):
            self.handle.remove()
        self.features = {}
        self.hook = self.get_features('hidden')
        self.handle = self.rnn.register_forward_hook(self.hook)

#%%

class BeliefRNNUnit(nn.Module):
    def __init__(self, input_size=10, hidden_size=10):
        super(BeliefRNNUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # linear weights
        self.input_w = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.transition_w = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.activation = nn.Sigmoid()
        
    def initialize_hidden_state(self):
        h = torch.randn(self.hidden_size)
        return F.softmax(h, dim=-1), h
        
    def process_input(self, x):
        """
        ≈ log(O @ o(t))
        basically if we observe a given observation,
        what log probabilities should we place on each state?
        note that sum(O * o(t)) = 1
        
        part of the sigmoid function looks like a log, so maybe this will work
        """
        return self.activation(self.input_w(x))
    
    def process_transition(self, b):
        """
        ≈ log(T @ b(t-1))
        """
        return self.activation(self.transition_w(b))

    def forward(self, x, tuple_in=None):
        if tuple_in is not None:
            (b_prev, h_prev) = tuple_in
        else:
            (b_prev, h_prev) = self.initialize_hidden_state()
        h = self.process_input(x) + self.process_transition(b_prev)
        b = F.softmax(h, dim=-1) # exponentiate and sum, so b is a probability
        return b, h
