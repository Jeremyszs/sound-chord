"""Bi-directional Transformer for Chord recognition (BTC).

Vendored from https://github.com/jayg996/BTC-ISMIR19 (btc_model.py and
utils/transformer_modules.py), Copyright (c) 2019 Jonggwon Park, MIT License.
See LICENSES/BTC-ISMIR19.txt.

Changes from upstream: merged into one module, inference-only forward pass that
returns logits, training/loss code removed, `np.float` replaced with `np.float64`.
Parameter names are unchanged so the original checkpoints load as-is.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gen_bias_mask(max_length: int) -> torch.Tensor:
    """-inf above the diagonal: masks future timesteps."""
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    return torch.from_numpy(np_mask).float().unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length: int, channels: int, min_timescale: float = 1.0,
                       max_timescale: float = 1.0e4) -> torch.Tensor:
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float64) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
    return torch.from_numpy(signal.reshape([1, length, channels])).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        # Registered as a buffer so it follows .to(device); not part of the checkpoint.
        if bias_mask is not None:
            self.register_buffer("bias_mask", bias_mask, persistent=False)
        else:
            self.bias_mask = None
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, t, d * h)

    def forward(self, queries, keys, values):
        queries = self._split_heads(self.query_linear(queries)) * self.query_scale
        keys = self._split_heads(self.key_linear(keys))
        values = self._split_heads(self.value_linear(values))
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if self.bias_mask is not None:
            logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]]
        weights = self.dropout(F.softmax(logits, dim=-1))
        return self.output_linear(self._merge_heads(torch.matmul(weights, values)))


class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super().__init__()
        padding = (kernel_size - 1, 0) if pad_type == "left" else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        return self.conv(self.pad(inputs.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    """conv + ReLU + conv (layer_config 'cc'), as used by BTC."""

    def __init__(self, input_depth, filter_size, output_depth, padding="both", dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv(input_depth, filter_size, kernel_size=3, pad_type=padding),
            Conv(filter_size, output_depth, kernel_size=3, pad_type=padding),
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Upstream applies ReLU after every layer, including the last one.
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads,
            bias_mask, attention_dropout)
        self.positionwise_convolution = PositionwiseFeedForward(
            hidden_size, filter_size, hidden_size, padding="both", dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, x):
        x_norm = self.layer_norm_mha(x)
        x = self.dropout(x + self.multi_head_attention(x_norm, x_norm, x_norm))
        x_norm = self.layer_norm_ffn(x)
        return self.dropout(x + self.positionwise_convolution(x_norm))


class BiDirectionalSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 max_length, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        common = (hidden_size, total_key_depth, total_value_depth, filter_size, num_heads)
        drops = (layer_dropout, attention_dropout, relu_dropout)
        mask = _gen_bias_mask(max_length)
        self.attn_block = SelfAttentionBlock(*common, mask, *drops)
        self.backward_attn_block = SelfAttentionBlock(*common, mask.transpose(2, 3), *drops)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))


class BiDirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth,
                 total_value_depth, filter_size, max_length=100, input_dropout=0.0,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        self.register_buffer("timing_signal", _gen_timing_signal(max_length, hidden_size),
                             persistent=False)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[
            BiDirectionalSelfAttention(hidden_size, total_key_depth, total_value_depth,
                                       filter_size, num_heads, max_length, layer_dropout,
                                       attention_dropout, relu_dropout)
            for _ in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        x = self.embedding_proj(self.input_dropout(inputs))
        x = x + self.timing_signal[:, :inputs.shape[1], :]
        return self.layer_norm(self.self_attn_layers(x))


class SoftmaxOutputLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, output_size)
        # Unused at inference, but present in the checkpoints.
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                            batch_first=True, bidirectional=True)

    def forward(self, hidden):
        return self.output_projection(hidden)


# Hyperparameters from upstream run_config.yaml. Only num_chords differs per checkpoint.
BTC_CONFIG = {
    "feature_size": 144,
    "timestep": 108,
    "num_layers": 8,
    "num_heads": 4,
    "hidden_size": 128,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
}


class BTCModel(nn.Module):
    def __init__(self, num_chords: int, config: dict | None = None):
        super().__init__()
        c = {**BTC_CONFIG, **(config or {})}
        self.timestep = c["timestep"]
        self.self_attn_layers = BiDirectionalSelfAttentionLayers(
            c["feature_size"], c["hidden_size"], c["num_layers"], c["num_heads"],
            c["total_key_depth"], c["total_value_depth"], c["filter_size"], c["timestep"])
        self.output_layer = SoftmaxOutputLayer(c["hidden_size"], num_chords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time <= 108, 144) normalised log-CQT. Returns logits (batch, time, chords)."""
        return self.output_layer(self.self_attn_layers(x))
