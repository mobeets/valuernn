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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
device = torch.device('cpu')

#%%

class ValueRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=1, 
                 num_layers=1, gamma=0.9, bias=True,
                 learn_weights=True, learn_initial_state=False,
                 predict_next_input=False, recurrent_cell='GRU', sigma_noise=0.0,
                 initialization_gain=None):
        super().__init__()

        self.gamma = gamma
        self.input_size = input_size # input dimensionality
        self.output_size = output_size # output dimensionality
        self.hidden_size = hidden_size # number of hidden recurrent units
        self.num_layers = num_layers # number of layers in recurrent layer; probably only works for 1
        self.recurrent_cell = recurrent_cell # type of recurrent cell
        self.predict_next_input = predict_next_input # not used
        self.sigma_noise = sigma_noise # s.d. of noise added internally to GRU

        # params used for initializing RNN
        self.kernel_initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        self.bias_regularizer = 'zeros'

        if recurrent_cell == 'GRU':
            if sigma_noise > 0:
                self.rnn = GRUWithNoise(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sigma_noise=sigma_noise)
            else:
                self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            
        else:
            assert sigma_noise == 0
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
        self.learn_weights = learn_weights
        
        self.bias = nn.Parameter(torch.tensor([0.0]*output_size))
        self.learn_bias = bias
        self.bias.requires_grad = self.learn_bias
        self.learn_initial_state = learn_initial_state
        if learn_initial_state:
            self.initial_state = nn.Parameter(torch.zeros(hidden_size))
            self.initial_state.requires_grad = learn_initial_state

        self.saved_weights = {}
        self.initialization_gain = initialization_gain
        self.reset(initialization_gain=self.initialization_gain)

    def forward(self, X, inactivation_indices=None, h0=None, return_hiddens=False, y=None, auto_readout_lr=0.0):
        """ v(t) = w.dot(z(t)), and z(t) = f(x(t), z(t-1)) """
        if inactivation_indices:
            return self.forward_with_lesion(X, inactivation_indices)
        
        # get initial state of RNN
        if h0 is None and self.learn_initial_state:
            if type(X) is torch.nn.utils.rnn.PackedSequence:
                batch_size = len(X[2])
            else:
                assert len(X.shape) == 3
                batch_size = X.shape[1]
            h0 = torch.tile(self.initial_state, (batch_size,1))[None,:]
        
        # pass inputs through RNN
        Z, last_hidden = self.rnn(X, hx=h0)
        
        if type(Z) is torch.nn.utils.rnn.PackedSequence:
            Z, _ = pad_packed_sequence(Z, batch_first=False)
        
        if return_hiddens:
            # only for GRU do we get full sequence of hiddens
            # because only in GRU are the hiddens the same as the outputs
            assert self.recurrent_cell == 'GRU'

        if y is None:
            V = self.bias + self.value(Z)
            hiddens = Z
        else:
            V, W = self.forward_with_td(Z, y, auto_readout_lr=auto_readout_lr)
            hiddens = (Z, W)

        return V, (hiddens if return_hiddens else last_hidden)
    
    def forward_with_td(self, Z, y, auto_readout_lr=0.0):
        """
        get value estimates, where the value readout is updated step-by-step using TD
        inputs:
        - Z: inputs with shape (seq_length, batch_size, hidden_size)
        - y: reward with shape (seq_length, batch_size, 1)
        - auto_readout_lr (float): learning rate for value weights
        outputs:
        - V: value estimate with shape (seq_length, batch_size, output_size)
        - W: readout weights with shape (seq_length, batch_size, hidden_size)

        n.b. model must be predicting r[t], not r[t+1]
        """
        assert self.recurrent_cell == 'GRU'
        vs = []
        ws = []
        w_t = torch.tile(self.value.weight, (Z.shape[1], 1)) # (batch_size, hidden_size)
        for t, z_t in enumerate(Z): # z_t is (batch_size, hidden_size)
            # get current value estimate
            w_t_prev = w_t # (batch_size, hidden_size)
            v_t = self.bias + (z_t * w_t_prev).sum(axis=-1) # (batch_size,)
            
            # update w_t using TD learning
            z_t_next = Z[t+1].detach() if t+1 < len(Z) else 0*z_t # (batch_size, hidden_size)
            v_t_next = self.bias.detach() + (z_t_next * w_t_prev.detach()).sum(axis=-1) # (batch_size,)
            d_t = y[t,:,0] + self.gamma*v_t_next - v_t.detach() # (batch_size,); n.b. x_t[:,-1] is r(t)
            w_t = w_t_prev + auto_readout_lr * d_t[:,None] * z_t.detach() # (batch_size, hidden_size)

            # save outputs
            ws.append(w_t_prev[None,:,:])
            vs.append(v_t)
        V = torch.vstack(vs)[:,:,None]
        W = torch.vstack(ws)
        return V, W
    
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
        for weight_ih, weight_hh, bias_ih, bias_hh in self.rnn.all_weights: # for each layer in rnn
            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)
            gate_inds = range(0, weight_hh.size(0), self.hidden_size)
            gate_names = ['reset', 'update', 'new']
            for name, i in zip(gate_names, gate_inds):
                nn.init.orthogonal_(weight_hh.data[i:(i+self.hidden_size)], gain=gain) # orthogonal
                nonlinearity = 'tanh' if ((self.recurrent_cell.lower() == 'rnn') or (name == 'new')) else 'sigmoid'
                nn.init.xavier_uniform_(weight_ih.data[i:(i+self.hidden_size)], gain=nn.init.calculate_gain(nonlinearity)) # glorot_uniform

    def reset(self, initialization_gain=None):
        self.bias.data *= 0
        if self.learn_initial_state:
            self.initial_state.data *= 0

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
        
class ValueSynapseRNN(ValueRNN):
    def __init__(self, representation_size=None, learn_representation=False, *args, **kwargs):
        self.representation_size = representation_size
        self.learn_representation = learn_representation
        super().__init__(*args, **kwargs)

        assert self.output_size == 1
        assert self.sigma_noise == 0
        assert self.recurrent_cell == 'GRU'
        assert self.learn_weights is True
        assert self.representation_size is not None
        assert self.recurrent_cell == 'GRU'

        self.representation = nn.Linear(in_features=self.input_size,
                                        out_features=self.representation_size,
                                        bias=True)
        self.representation.weight.requires_grad = self.learn_representation
        self.representation.bias.requires_grad = self.learn_representation

        self.rnn = nn.GRU(input_size=self.representation_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
        self.initial_state = nn.Parameter(torch.zeros(self.hidden_size))
        self.initial_state.requires_grad = self.learn_initial_state
        self.value = nn.Linear(in_features=self.hidden_size,
                               out_features=self.representation_size,
                               bias=True)

        self.reset(initialization_gain=self.initialization_gain)

    def forward(self, xin, inactivation_indices=None, h0=None, return_hiddens=False):
        """ v(t) = w(t).dot(x(t)), and w(t) = f(x(t-1), w(t-1)) """
        assert inactivation_indices is None
        wasPacked = type(xin) is torch.nn.utils.rnn.PackedSequence
        if wasPacked:
            xin, x_lengths = pad_packed_sequence(xin, batch_first=False)
        x = self.representation(xin) # ignore reward here

        xinc = torch.clone(xin); xinc[:,:,-1] = 0; # remove reward
        x_orig = self.representation(xinc) # representation ignoring reward

        h0 = torch.tile(torch.unsqueeze(torch.unsqueeze(self.initial_state, 0), 0), (1,x.shape[1],1)) if h0 is None else h0
        # x_orig = x
        if wasPacked:
            x = pack_padded_sequence(x, x_lengths, enforce_sorted=False)
        w, hidden = self.rnn(x, hx=h0) # n.b. assumes x contains current reward
        if type(w) is torch.nn.utils.rnn.PackedSequence:
            w, _ = pad_packed_sequence(w, batch_first=False)
        w = self.value(torch.vstack([h0, w]))
        v = torch.einsum('ijk,ijk->ij', x_orig, w[:-1]).unsqueeze(2)
        if return_hiddens:
            # only for GRU do we get full sequence of hiddens
            # because only in GRU are the hiddens the same as the outputs
            assert self.recurrent_cell == 'GRU'
        return self.bias + v, (w if return_hiddens else hidden)
    
    def reset(self, initialization_gain=None):
        super().reset(initialization_gain=initialization_gain)
        if hasattr(self, 'representation'): # if not, we are in parent constructor
            self.representation.bias.data *= 0
            self.value.bias.data *= 0
        self.initial_weights = self.checkpoint_weights()

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
