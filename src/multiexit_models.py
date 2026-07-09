#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiexit_models.py -- multi-exit backbone definitions for MT-SFL and the baselines.

Defines the per-tier backbone blocks phi_{k,l} (GRU or Transformer) and the early-exit auxiliary
classifiers beta_{k,l}. Each upper tier continues from the activation forwarded by the tier below,
following the split-inference procedure described in the paper.
"""
import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------ common input adapter
class _InputAdapter(nn.Module):
    """Build (B,T,d) via either token embedding (vocab) or a linear projection of continuous features (in_dim)."""
    def __init__(self, d, vocab_size=None, in_dim=None, pad_id=None):
        super().__init__()
        assert (vocab_size is not None) ^ (in_dim is not None), \
            "specify exactly one of vocab_size or in_dim"
        self.pad_id = pad_id
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_id)
            self.proj = None
        else:
            self.embed = None
            self.proj = nn.Linear(in_dim, d)

    def forward(self, x):
        return self.embed(x) if self.embed is not None else self.proj(x)

    def pad_mask(self, x):
        """key_padding_mask(B,T): True=pad. Meaningful only for token inputs (based on pad_id)."""
        if self.embed is not None and self.pad_id is not None:
            return x == self.pad_id
        return None


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _n(module):
    return sum(p.numel() for p in module.parameters())


# ====================================================================== (A) uniform-width GRU
class MultiExitGRU(nn.Module):
    """4-tier GRU with uniform width D. tier m = GRU layer(s) + early-exit head (last step)."""
    def __init__(self, num_classes, dim=256, n_layers_per_tier=1, side_dim=0,
                 vocab_size=None, in_dim=None, pad_id=None, dropout=0.1):
        super().__init__()
        self.adapter = _InputAdapter(dim, vocab_size, in_dim, pad_id)
        self.tiers = nn.ModuleList([
            nn.GRU(dim, dim, num_layers=n_layers_per_tier, batch_first=True)
            for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(dim + side_dim, num_classes) for _ in range(4)])
        self.n_exits = 4

    def forward(self, x, side=None, pad_mask=None):
        h = self.adapter(x)                       # (B,T,D)
        out = []
        for m in range(4):
            h, _ = self.tiers[m](h)               # (B,T,D)
            summ = h[:, -1, :]                    # last timestep (with left-padding, the most recent real token)
            z = torch.cat([summ, side], 1) if (side is not None and side.size(1)) else summ
            out.append(self.heads[m](self.drop(z)))
        return out

    def exit_param_counts(self):
        """Cumulative parameters on the exit-m inference path = adapter + tiers[0..m] + heads[m]."""
        base = _n(self.adapter)
        cum, counts = base, []
        for m in range(4):
            cum += _n(self.tiers[m])
            counts.append(cum + _n(self.heads[m]))
        return counts


# =============================================================== (B) width-increasing Transformer
class MultiExitTransformer(nn.Module):
    """Self-attention backbone whose d_model grows with tier. Width is expanded by a linear projection between tiers."""
    def __init__(self, num_classes, dims=(96, 160, 256, 384), nhead=4, n_layers_per_tier=1,
                 ff_mult=4, side_dim=0, vocab_size=None, in_dim=None, pad_id=None, dropout=0.1):
        super().__init__()
        assert len(dims) == 4, "dims are the widths of the 4 tiers"
        for d in dims:
            assert d % nhead == 0, f"d_model({d}) must be divisible by nhead({nhead})"
        self.dims = tuple(dims)
        self.adapter = _InputAdapter(dims[0], vocab_size, in_dim, pad_id)
        self.pos = PositionalEncoding(dims[0])
        self.tier_proj = nn.ModuleList()
        self.tier_blocks = nn.ModuleList()
        prev = dims[0]
        for d in dims:
            self.tier_proj.append(nn.Linear(prev, d) if prev != d else nn.Identity())
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=d * ff_mult,
                dropout=dropout, batch_first=True, activation="gelu")
            self.tier_blocks.append(nn.TransformerEncoder(layer, num_layers=n_layers_per_tier))
            prev = d
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d + side_dim, num_classes) for d in dims])
        self.n_exits = 4

    def forward(self, x, side=None, pad_mask=None):
        if pad_mask is None:
            pad_mask = self.adapter.pad_mask(x)
        h = self.pos(self.adapter(x))             # (B,T,d0)
        out = []
        for i in range(4):
            h = self.tier_proj[i](h)              # width expansion
            h = self.tier_blocks[i](h, src_key_padding_mask=pad_mask)
            summ = h[:, -1, :]                    # last token (with left-padding, the most recent real token)
            z = torch.cat([summ, side], 1) if (side is not None and side.size(1)) else summ
            out.append(self.heads[i](self.drop(z)))
        return out

    def exit_param_counts(self):
        base = _n(self.adapter)                    # PositionalEncoding is a buffer (0 params)
        cum, counts = base, []
        for i in range(4):
            cum += _n(self.tier_proj[i]) + _n(self.tier_blocks[i])
            counts.append(cum + _n(self.heads[i]))
        return counts


def build_model(arch, num_classes, side_dim=0, vocab_size=None, in_dim=None,
                pad_id=None, dropout=0.1, dim=256, dims=(96, 160, 256, 384),
                nhead=4, n_layers_per_tier=1):
    if arch == "gru":
        return MultiExitGRU(num_classes, dim=dim, n_layers_per_tier=n_layers_per_tier,
                            side_dim=side_dim, vocab_size=vocab_size, in_dim=in_dim,
                            pad_id=pad_id, dropout=dropout)
    elif arch == "transformer":
        return MultiExitTransformer(num_classes, dims=tuple(dims), nhead=nhead,
                                    n_layers_per_tier=n_layers_per_tier, side_dim=side_dim,
                                    vocab_size=vocab_size, in_dim=in_dim, pad_id=pad_id,
                                    dropout=dropout)
    raise ValueError(f"unknown arch: {arch}")
