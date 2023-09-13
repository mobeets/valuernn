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
device = torch.device('cpu')

#%%

class ValueRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=1, 
                 num_layers=1, gamma=0.9, bias=True, learn_weights=True, predict_next_input=False,
                 recurrent_cell='GRU', sigma_noise=0.0, initialization_gain=None):
        super(ValueRNN, self).__init__()

        self.gamma = gamma
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_cell = recurrent_cell
        self.kernel_initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        self.bias_regularizer = 'zeros'
        self.predict_next_input = predict_next_input

        if recurrent_cell == 'GRU':
            if sigma_noise > 0:
                self.rnn = GRUWithNoise(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sigma_noise=sigma_noise)
            else:
                self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            
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
            self.value = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        else:
            self.value = lambda x: torch.sum(x,2)[:,:,None]
        
        self.bias = nn.Parameter(torch.tensor([0.0]*output_size))
        self.bias.requires_grad = bias
        self.saved_weights = {}
        self.initialization_gain = initialization_gain
        self.reset(initialization_gain=self.initialization_gain)

    def forward(self, xin, inactivation_indices=None):
        if inactivation_indices:
            return self.forward_with_lesion(xin, inactivation_indices)
        x, hidden = self.rnn(xin)
        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x, output_lengths = pad_packed_sequence(x, batch_first=False)
        return self.bias + self.value(x), hidden
    
    def forward_with_lesion(self, x, indices=None):
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

    def reset(self, initialization_gain=None):
        self.bias = nn.Parameter(torch.tensor(0.0))
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        if initialization_gain is not None and initialization_gain != 0:
            self.initialize(initialization_gain)
        self.initial_weights = self.checkpoint_weights()
               
    def checkpoint_weights(self):
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

from torch.nn import RNNBase
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor
from typing import Tuple, Optional, overload
from torch import _VF

class GRUWithNoise(RNNBase):
    def __init__(self, *args, **kwargs):
        if 'sigma_noise' in kwargs:
            self.sigma_noise = kwargs.pop('sigma_noise')
        else:
            self.sigma_noise = 0.0
        super(GRUWithNoise, self).__init__('GRU', *args, **kwargs)

    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:  # noqa: F811
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:  # noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        
        assert self.num_layers == 1, "Only one layer implemented"
        hx = hx[0,:,:]
        
        if batch_sizes is None:
            assert is_batched, "Only implemented for batched inputs"
            assert not self.batch_first, "Not implemented for batch_first"
            output = torch.zeros(
                input.size(0), input.size(1), self.hidden_size,
                dtype=input.dtype, device=input.device)
            for seq_index in range(input.size(0)):
                new_hx = _VF.gru_cell(
                    input[seq_index,:,:],
                    hx, self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0, self.bias_hh_l0)
                if self.sigma_noise > 0:
                    # add noise to hidden units
                    new_hx += self.sigma_noise * torch.randn_like(new_hx)
                output[seq_index,:,:] = new_hx
                hx = new_hx
        else:
            output = torch.zeros(
                input.size(0), self.hidden_size,
                dtype=input.dtype, device=input.device)
            begin = 0
            for batch in batch_sizes:
                new_hx = _VF.gru_cell(
                    input[begin: begin + batch],
                    hx[0:batch], self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0, self.bias_hh_l0)
                if self.sigma_noise > 0:
                    # add noise to hidden units
                    new_hx += self.sigma_noise * torch.randn_like(new_hx)
                output[begin: begin + batch] = new_hx
                hx = new_hx
                begin += batch

        hidden = [] # I don't think we use this so who cares

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, [] #self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                # hidden = hidden.squeeze(1)
            return output, []#self.permute_hidden(hidden, unsorted_indices)
