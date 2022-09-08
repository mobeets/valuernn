#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:33:12 2022

@author: mobeets
"""

import torch
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

#%%

doTestBatchSequence = False
doTestPackedSequence = False

if doTestBatchSequence or doTestPackedSequence:
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.utils.rnn import pack_padded_sequence
    input_size = 2
    hidden_size = 4
    seq_length = 15
    batch_size = 5

if doTestBatchSequence:
    rnn = GRUWithNoise(input_size, hidden_size, sigma_noise=0.01)
    input = torch.randn(seq_length, batch_size, rnn.input_size)
    h0 = torch.randn(1, batch_size, rnn.hidden_size)
    output, hn = rnn(input, h0)
    rnn.sigma_noise = 0.0
    output2, hn = rnn(input)
    
    z1 = output.detach().numpy()
    z2 = output2.detach().numpy()
    
if doTestPackedSequence:
    rnn = GRUWithNoise(input_size, hidden_size, sigma_noise=0.01)
    input = torch.randn(seq_length, batch_size, rnn.input_size)
    input = pack_padded_sequence(input, [seq_length]*batch_size, enforce_sorted=False)
    h0 = torch.randn(1, batch_size, rnn.hidden_size)
    output, hn = rnn(input, h0)
    rnn.sigma_noise = 0.0
    output2, hn = rnn(input)
    
    z1 = output[0].detach().numpy()
    z2 = output2[0].detach().numpy()
    z1 = z1.reshape((seq_length, batch_size, hidden_size))
    z2 = z2.reshape((seq_length, batch_size, hidden_size))
    
if doTestBatchSequence or doTestPackedSequence:
    xs = np.arange(seq_length)
    for c in range(hidden_size):
        for b in range(batch_size):
            h = plt.plot(b*(seq_length+1) + xs, z1[:,b,c] + c, '.-', markersize=2)
            plt.plot(b*(seq_length+1) + xs, z2[:,b,c] + c, '.-', markersize=2, color=h[0].get_color(), alpha=0.5)
