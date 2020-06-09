#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:04:17 2020

@author: krishna
"""
import torch
import torch.nn as nn

class Convolution_Block(nn.Module):
    def __init__(self, input_dim=40,cnn_out_channels=64):
        super(Convolution_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, stride=1,padding=3),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels,cnn_out_channels, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        out = self.conv(inputs)
        return out


class DropSubsampler(nn.Module):
    """Subsample by droping input frames."""

    def __init__(self, factor):
        super(DropSubsampler, self).__init__()

        self.factor = factor

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        xs = xs[:, ::self.factor, :]

        xlens = [max(1, (i + self.factor - 1) // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating successive input frames."""

    def __init__(self, factor, n_units):
        super(ConcatSubsampler, self).__init__()

        self.factor = factor
        if factor > 1:
            self.proj = nn.Linear(n_units * factor, n_units)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        xs = xs.transpose(1, 0).contiguous()
        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1, -1, -1)], dim=-1)
              for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        xs = torch.cat(xs, dim=0).transpose(1, 0)
        # NOTE: Exclude the last frames if the length is not divisible

        xs = torch.relu(self.proj(xs))
        xlens //= self.factor
        return xs, xlens





class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1,dropout=0.3,cnn_out_channels=64,rnn_celltype='gru'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout
      
        self.conv = Convolution_Block(self.input_dim,cnn_out_channels=cnn_out_channels)
        
        if rnn_celltype == 'lstm':
            self.rnn =  nn.LSTM(cnn_out_channels, self.hidden_dim, self.n_layers, dropout=self.dropout_p, bidirectional=False,batch_first=True)
        else:
            self.rnn =  nn.GRU(cnn_out_channels, self.hidden_dim, self.n_layers, dropout=self.dropout_p, bidirectional=False,batch_first=True)

    def forward(self, inputs, input_lengths):
        
        output_lengths = self.get_conv_out_lens(input_lengths)
        out = self.conv(inputs) 
        out = out.permute(0,2,1)
        
        out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, enforce_sorted=False, batch_first=True)
        out, rnn_hidden_state = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        rnn_out = out.transpose(0, 1)
        
        return rnn_out, rnn_hidden_state


    def get_conv_out_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv1d :
                seq_len = ((seq_len + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) / m.stride[0] + 1)

        return seq_len.int()
