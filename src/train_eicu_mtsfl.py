#!/usr/bin/env python3
"""
train_eicu_mtsfl.py -- main training / evaluation entry point for the eICU experiments.

Implements MT-SFL (Multi-Tier Split Federated Learning): a four-tier (client / edge / region /
cloud) split-FL model with tier-wise FedAvg aggregation at periods tau = [tau2, tau3, tau4],
per-tier personalization weights lambda = [lambda1, lambda2, lambda3], multi-exit auxiliary
classifiers weighted by gamma, and Shannon-entropy early-exit routing at fixed per-exit
thresholds E_th. The six comparison baselines are selected with --baseline (see baselines_eicu.py).

Task/metric: 25-label ICU phenotype multi-label classification; macro-AUROC over the 25 phenotypes.
MT-SFL is scored on routed, label-wise predictions (each (sample, phenotype) is read from the tier
where the entropy criterion is first met); baselines are scored on their single full model.

Key arguments: --rnn {gru,transformer} --hidden ... --gamma g1 g2 g3 --lambda1/2/3 ...
  --tau2 --tau3 --tau4 --lr --weight_decay --lr_sched cosine --num_epochs --local_epochs
  --test_mode {temporal,labelshift} --eth_fixed "e1,e2,e3" --tune_eth --seed --run_tag
See scripts/run_mtsfl.sh and scripts/run_baselines.sh for the exact commands reproducing the paper.
"""

import argparse
import csv
import json
import math
import os
import copy
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

N_CH = 8  # 4 vital + 4 mask


def vital_summaries(X, Mk):
    """Per-channel summary statistics over observed values (mask=1): [last, mean, min, max, last-first] x C channels.
    Lets each exit head read the vitals directly, separately from the sequence model, to strengthen the input signal (same as central)."""
    N, T, C = X.shape
    obs = Mk > 0.5
    cnt = obs.sum(axis=1)
    safe = np.maximum(cnt, 1)
    mean = (X * obs).sum(axis=1) / safe
    Xmin = np.where(cnt > 0, np.where(obs, X, np.inf).min(axis=1), 0.0)
    Xmax = np.where(cnt > 0, np.where(obs, X, -np.inf).max(axis=1), 0.0)
    arange = np.arange(T)[None, :, None]
    first_idx = np.where(obs, arange, T).min(axis=1)
    last_idx = np.where(obs, arange, -1).max(axis=1)
    fi = np.clip(first_idx, 0, T - 1)[:, None, :]
    li = np.clip(last_idx, 0, T - 1)[:, None, :]
    first_val = np.where(cnt > 0, np.take_along_axis(X, fi, axis=1)[:, 0, :], 0.0)
    last_val = np.where(cnt > 0, np.take_along_axis(X, li, axis=1)[:, 0, :], 0.0)
    delta = last_val - first_val
    return np.concatenate([last_val, mean, Xmin, Xmax, delta], axis=1).astype(np.float32)  # (N,5C)


# ====================================================================
# 1. 4-tier RNN (GRU/LSTM) backbone -- hidden size (parameters) grows from client to cloud.
#    the hidden-state sequence is passed between tiers; early-exit at each tier's last step + lab.
# ====================================================================
class TierRNN(nn.Module):
    """One tier: RNN -> (output sequence, aux logits). The final tier returns only log_softmax."""
    def __init__(self, in_dim, hidden, lab_dim, n_classes, rnn="gru",
                 bidir=False, dropout=0.0, transpose_in=False, final=False, multilabel=False):
        super().__init__()
        self.transpose_in = transpose_in          # client tier: (B,C,T)->(B,T,C)
        self.final = final
        self.multilabel = multilabel              # if True, the final tier also returns raw logits (BCE/sigmoid)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(in_dim, hidden, batch_first=True, bidirectional=bidir)
        self.out_dim = hidden * (2 if bidir else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.out_dim + lab_dim, n_classes)

    def forward(self, x, lab=None):
        if self.transpose_in:
            x = x.transpose(1, 2)                  # (B,C,T) -> (B,T,C)
        out, _ = self.rnn(x)                       # (B,T,out_dim)
        summ = out[:, -1, :]                       # last-timestep summary
        z = torch.cat([summ, lab], dim=1) if lab is not None and lab.size(1) else summ
        logits = self.head(self.drop(z))
        if self.final:
            return logits if self.multilabel else F.log_softmax(logits, dim=1)
        return out, logits


class TierTransformer(nn.Module):
    """One tier: input projection + positional embedding -> TransformerEncoder -> (output sequence, aux logits).
    Same interface as TierRNN. The final tier returns (multilabel? raw logits : log_softmax).
    The sequence summary is mean-pool (more natural than last-step for a transformer). out_dim=d_model."""
    def __init__(self, in_dim, d_model, lab_dim, n_classes, nhead=4, ff_mult=4,
                 dropout=0.0, transpose_in=False, final=False, multilabel=False,
                 n_layers=1, max_len=512):
        super().__init__()
        self.transpose_in = transpose_in
        self.final = final
        self.multilabel = multilabel
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_mult, dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out_dim = d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model + lab_dim, n_classes)

    def forward(self, x, lab=None):
        if self.transpose_in:
            x = x.transpose(1, 2)                  # (B,C,T) -> (B,T,C)
        h = self.inp(x) + self.pos[:, :x.size(1), :]
        out = self.enc(h)                          # (B,T,d_model)
        summ = out.mean(dim=1)                     # mean-pool summary
        z = torch.cat([summ, lab], dim=1) if lab is not None and lab.size(1) else summ
        logits = self.head(self.drop(z))
        if self.final:
            return logits if self.multilabel else F.log_softmax(logits, dim=1)
        return out, logits


def build_tiers(n_classes, lab_dim, hidden, rnn="gru", bidir=False, dropout=0.0, n_ch=N_CH,
                multilabel=False, nhead=4, ff_mult=4):
    """hidden=[H_client,H_edge,H_region,H_cloud]. Returns the 4 tier modules. n_ch = number of backbone input channels.
    If rnn='transformer', a Transformer encoder per tier (width=hidden[tier]); else GRU/LSTM."""
    if rnn == "transformer":
        client = TierTransformer(n_ch, hidden[0], lab_dim, n_classes, nhead, ff_mult, dropout, transpose_in=True)
        edge = TierTransformer(hidden[0], hidden[1], lab_dim, n_classes, nhead, ff_mult, dropout)
        region = TierTransformer(hidden[1], hidden[2], lab_dim, n_classes, nhead, ff_mult, dropout)
        cloud = TierTransformer(hidden[2], hidden[3], lab_dim, n_classes, nhead, ff_mult, dropout,
                                final=True, multilabel=multilabel)
        return client, edge, region, cloud
    d = 2 if bidir else 1
    client = TierRNN(n_ch, hidden[0], lab_dim, n_classes, rnn, bidir, dropout, transpose_in=True)
    edge = TierRNN(hidden[0] * d, hidden[1], lab_dim, n_classes, rnn, bidir, dropout)
    region = TierRNN(hidden[1] * d, hidden[2], lab_dim, n_classes, rnn, bidir, dropout)
    cloud = TierRNN(hidden[2] * d, hidden[3], lab_dim, n_classes, rnn, bidir, dropout,
                    final=True, multilabel=multilabel)
    return client, edge, region, cloud


class SingleEICUModel(nn.Module):
    """A plain single model with no exits/tiers (only for probing eICU time-series learnability; not a paper baseline).
    Same input pipeline: (B,C,T) time series -> RNN/TF -> + lab(side) concat -> a single head.
    phenotype=multilabel(raw logits/BCE), diagnosis=log_softmax. GRU/LSTM: num_layers deep. TF: num_layers encoder + mean-pool."""
    def __init__(self, n_ch, hidden, lab_dim, n_classes, rnn="gru", dropout=0.0,
                 num_layers=2, nhead=4, ff_mult=4, multilabel=True, max_len=512):
        super().__init__()
        self.rnn_type = rnn
        self.multilabel = multilabel
        if rnn == "transformer":
            self.inp = nn.Linear(n_ch, hidden)
            self.pos = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(hidden, nhead, hidden * ff_mult, dropout,
                                               batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers)
        else:
            RNN = nn.LSTM if rnn == "lstm" else nn.GRU
            self.rnn = RNN(n_ch, hidden, num_layers=num_layers, batch_first=True,
                           dropout=(dropout if num_layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + lab_dim, n_classes)

    def forward(self, x, lab=None):                        # x:(B,C,T), lab:(B,S)
        x = x.transpose(1, 2)                              # (B,C,T) -> (B,T,C)
        if self.rnn_type == "transformer":
            h = self.inp(x) + self.pos[:, :x.size(1), :]
            z = self.enc(h).mean(dim=1)
        else:
            out, _ = self.rnn(x)
            z = out[:, -1, :]
        if lab is not None and lab.size(1):
            z = torch.cat([z, lab], dim=1)
        logits = self.head(self.drop(z))
        return logits if self.multilabel else F.log_softmax(logits, dim=1)


class FullClientModel:
    """A full 4-tier copy for a client (MT-SFL: a per-client copy + periodic synchronization)."""
    def __init__(self, n_classes, device, lab_dim=0, dropout=0.0,
                 hidden=(256, 256, 256, 256), rnn="gru", bidir=False, n_ch=N_CH, multilabel=False,
                 nhead=4, ff_mult=4):
        c, e, r, cl = build_tiers(n_classes, lab_dim, hidden, rnn, bidir, dropout, n_ch=n_ch,
                                  multilabel=multilabel, nhead=nhead, ff_mult=ff_mult)
        self.client, self.edge, self.region, self.cloud = (
            c.to(device), e.to(device), r.to(device), cl.to(device))

    def parameters(self):
        return (list(self.client.parameters()) + list(self.edge.parameters()) +
                list(self.region.parameters()) + list(self.cloud.parameters()))

    def train_mode(self):
        for m in (self.client, self.edge, self.region, self.cloud):
            m.train()

    def eval_mode(self):
        for m in (self.client, self.edge, self.region, self.cloud):
            m.eval()


# ====================================================================
# 2. state_dict arithmetic (hierarchical FedAvg + lambda blending). Integer BN buffers keep the first model's values.
# ====================================================================
def get_sd(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def sd_mean(sds):
    out = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            out[k] = torch.stack([sd[k] for sd in sds]).mean(0)
        else:
            out[k] = sds[0][k].clone()
    return out


def sd_blend(a, b, lam):
    """a(personal) <- lam*a + (1-lam)*b(aggregated). Integer buffers keep a."""
    out = {}
    for k in a:
        out[k] = lam * a[k] + (1 - lam) * b[k] if a[k].is_floating_point() else a[k].clone()
    return out


def submodels_mean(models):
    return sd_mean([get_sd(m) for m in models])


def weighted_sd_mean(sds, weights):
    """|D_k|-weighted FedAvg. weights is a 1-D list that sums to 1."""
    out = {}
    dev = next(iter(sds[0].values())).device          # device of the model parameters (cuda/cpu)
    w0 = torch.tensor(weights, dtype=torch.float32, device=dev)
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            stacked = torch.stack([sd[k] for sd in sds])           # (M, ...)
            w = w0.to(stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
            out[k] = (stacked * w).sum(0)
        else:
            out[k] = sds[0][k].clone()
    return out


def weighted_block_mean(clients, members, attr, counts):
    """|D_k|-weighted average of the block (attr) across the member clients."""
    sds = [get_sd(getattr(clients[c], attr)) for c in members]
    w = np.array([counts[c] for c in members], dtype=float)
    w = w / w.sum()
    return weighted_sd_mean(sds, w.tolist())


# ====================================================================
# 3. derive the scope labels (T_{k,l}): train-prevalence top-M set-difference
# ====================================================================
def topm(counter, m):
    return set(l for l, _ in counter.most_common(m))


def derive_scope_labels(meta_tr, n_clients, client_of, hosp_of, region_of,
                        hosp2clients, region2clients, top_m):
    """top_m = (M_client, M_edge, M_region), an increasing M per tier. An int / single element applies to all three tiers.
    T4(outside) = all labels not in client/edge/region (= generalization; absorbs the old outside)."""
    if isinstance(top_m, (int, float)):
        m1 = m2 = m3 = int(top_m)
    elif len(top_m) == 1:
        m1 = m2 = m3 = int(top_m[0])
    else:
        m1, m2, m3 = int(top_m[0]), int(top_m[1]), int(top_m[2])
    g = Counter(meta_tr["y"])
    all_labels = set(g.keys())
    by_client = {c: Counter() for c in range(n_clients)}
    by_hosp = defaultdict(Counter)
    by_region = defaultdict(Counter)
    for c, y in zip(meta_tr["client_idx"], meta_tr["y"]):
        by_client[c][y] += 1
        by_hosp[hosp_of[c]][y] += 1
        by_region[region_of[c]][y] += 1
    scope = {}
    for c in range(n_clients):
        cl = topm(by_client[c], m1)
        ed = topm(by_hosp[hosp_of[c]], m2) - cl
        rg = topm(by_region[region_of[c]], m3) - cl - ed
        out = all_labels - cl - ed - rg          # T4 = everything else (absorbs outside)
        scope[c] = {"client": cl, "edge": ed, "region": rg, "outside": out}
    return scope


# ====================================================================
# 4. training / inference
# ====================================================================
def client_train(full, loader, opt, gamma, device, max_batches=0, class_weight=None, multilabel=False):
    full.train_mode()
    for bi, (x, lab, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        if x.size(0) < 2:   # skip tiny batches (stability)
            continue
        x, lab, y = x.to(device), lab.to(device), y.to(device)
        c_out, c_aux = full.client(x, lab)
        e_out, e_aux = full.edge(c_out, lab)
        r_out, r_aux = full.region(e_out, lab)
        cl_out = full.cloud(r_out, lab)
        if multilabel:      # phenotype: raw logits at every exit + BCE (cl_out is raw logits too)
            loss = (gamma[0] * F.binary_cross_entropy_with_logits(c_aux, y) +
                    gamma[1] * F.binary_cross_entropy_with_logits(e_aux, y) +
                    gamma[2] * F.binary_cross_entropy_with_logits(r_aux, y) +
                    (1 - sum(gamma)) * F.binary_cross_entropy_with_logits(cl_out, y))
        else:               # diagnosis: aux=CE, cloud=log_softmax+nll
            loss = (gamma[0] * F.cross_entropy(c_aux, y, weight=class_weight) +
                    gamma[1] * F.cross_entropy(e_aux, y, weight=class_weight) +
                    gamma[2] * F.cross_entropy(r_aux, y, weight=class_weight) +
                    (1 - sum(gamma)) * F.nll_loss(cl_out, y, weight=class_weight))
        opt.zero_grad()
        loss.backward()
        opt.step()


def entropy(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-10)).sum(1)


def mean_binary_entropy(logits):
    """multi-label(phenotype) confidence: mean of the per-label binary entropy after sigmoid.
    Lower value means more confident -> early-exit. Return shape=(B,)."""
    p = torch.sigmoid(logits)
    h = -(p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10))
    return h.mean(dim=1)


def infer_with_exits(full, x, lab, e1, e2, e3):
    """entropy early-exit. Returns: preds, tier(0=client..3=cloud) per sample."""
    n = x.size(0)
    dev = x.device
    preds = torch.zeros(n, dtype=torch.long, device=dev)
    tier = torch.full((n,), -1, dtype=torch.long, device=dev)

    c_out, c_aux = full.client(x, lab)
    m1 = entropy(c_aux) < e1
    if m1.any():
        i = m1.nonzero(as_tuple=False).view(-1)
        preds[i] = c_aux[i].argmax(1); tier[i] = 0
    rest = (~m1).nonzero(as_tuple=False).view(-1)
    if rest.numel():
        e_out, e_aux = full.edge(c_out[rest], lab[rest])
        m2 = entropy(e_aux) < e2
        if m2.any():
            i = rest[m2]; preds[i] = e_aux[m2].argmax(1); tier[i] = 1
        rest2 = rest[~m2]
        if rest2.numel():
            r_out, r_aux = full.region(e_out[~m2], lab[rest2])
            m3 = entropy(r_aux) < e3
            if m3.any():
                i = rest2[m3]; preds[i] = r_aux[m3].argmax(1); tier[i] = 2
            rest3 = rest2[~m3]
            if rest3.numel():
                cl_out = full.cloud(r_out[~m3], lab[rest3])
                preds[rest3] = cl_out.argmax(1); tier[rest3] = 3
    return preds, tier


# ====================================================================
# 5. hierarchical aggregation -- faithful implementation of paper Eqs. (5)-(10).
#    upward: |D_k|-weighted FedAvg of block u at tier l (2..m) (Eqs. 5-7; via weighted telescoping,
#          equivalent to directly weighting the descendant clients). downward: lambda blend from tier m->1 (Eqs. 8-9).
#    final value of block u for each client = Sum_{l} (nested-lambda coefficient) * phi^(l)_{anc_l(k),u}.
#    e.g. u=1,m=4: lam1*local + (1-lam1)lam2*hosp + (1-lam1)(1-lam2)lam3*region + (1-lam1)(1-lam2)(1-lam3)*global.
#    per-tier period E_m: each epoch, only the highest triggering tier m* cascades (including lower tiers).
#    cold clients (0 train samples) are excluded from the average and only receive.
# ====================================================================
ATTR = {1: "client", 2: "edge", 3: "region", 4: "cloud"}


def aggregate(clients, m, members_by, anc, counts, lambdas):
    """tier-m cascade aggregation. members_by[l]:node->clients, anc[l]:client->node (l=2..4)."""
    n = len(clients)
    for u in range(1, m + 1):
        a = ATTR[u]
        # upward: |D|-weighted average only over the tiers that block u actually uses (l=max(2,u)..m)
        tier_avg = {}
        for l in range(max(2, u), m + 1):
            tier_avg[l] = {}
            for node, mem in members_by[l].items():
                mem2 = [c for c in mem if counts[c] > 0]
                if mem2:
                    tier_avg[l][node] = weighted_block_mean(clients, mem2, a, counts)
        # downward: lambda blend from tier m->u (or 1) for each client
        new_sd = {}
        low = 1 if u == 1 else u
        for c in range(n):
            top_node = anc[m][c]
            if top_node not in tier_avg[m]:        # if the whole top group is cold, skip (keep local)
                continue
            cur = tier_avg[m][top_node]            # phi_tilde^(m) = phi^(m) (top, no blend)
            for l in range(m - 1, low - 1, -1):
                lam = lambdas[l - 1]
                if l == 1:
                    local_l = get_sd(getattr(clients[c], a))
                else:
                    node_l = anc[l][c]
                    if node_l not in tier_avg[l]:  # whole group cold -> skip this tier term (keep upper value)
                        continue
                    local_l = tier_avg[l][node_l]
                cur = sd_blend(local_l, cur, lam)  # λ_l·local_l + (1-λ_l)·cur
            new_sd[c] = cur
        for c, sd in new_sd.items():
            getattr(clients[c], a).load_state_dict(sd)


# ====================================================================
# 6. main
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="mtsfl_eicu")
    ap.add_argument("--hospital_csv", default=None,
                    help="hospital.csv for the region join (default: search data_dir or its parent)")
    ap.add_argument("--num_epochs", type=int, default=120)
    ap.add_argument("--tau2", type=int, default=3)
    ap.add_argument("--tau3", type=int, default=6)
    ap.add_argument("--tau4", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                    help="adam (lr~1e-3) recommended -- resolves stagnation")
    ap.add_argument("--local_epochs", type=int, default=1, help="number of local client passes per round")
    ap.add_argument("--dropout", type=float, default=0.3, help="classifier head dropout (curbs overfitting)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--rnn", choices=["gru", "lstm", "transformer"], default="gru")
    ap.add_argument("--nhead", type=int, default=4, help="number of transformer heads (--rnn transformer)")
    ap.add_argument("--ff_mult", type=int, default=4, help="transformer FFN multiplier")
    ap.add_argument("--hidden", type=int, nargs=4, default=[256, 256, 256, 256],
                    help="per-tier hidden [client edge region cloud]. GRU=uniform (equal-size blocks)")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--time_pool", type=int, default=3,
                    help="temporal mean-pool factor (288/k). 3=96-step (matches central). 1=original resolution")
    ap.add_argument("--no_agg", action="store_true",
                    help="disable all hierarchical aggregation = local-only baseline (no collaboration)")
    ap.add_argument("--centralized", action="store_true",
                    help="centralized baseline for comparison: train the whole set as a single shared model (no aggregation). evaluation is per-client")
    ap.add_argument("--val_frac", type=float, default=0.0,
                    help=">0 splits a patient-level val out of train and evaluates on it (for tuning only). test is not used")
    ap.add_argument("--run_tag", default="", help="tag for the result JSON filename (tune_<tag>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra_channels", action="store_true",
                    help="concat eicu_X_extra.npy (NIBP x3 + GCS, etc.) to the backbone")
    ap.add_argument("--extra_suffix", default="", help="extra file suffix (e.g. _all)")
    ap.add_argument("--task", choices=["diagnosis", "phenotype"], default="diagnosis",
                    help="diagnosis=single-label (CE/acc/scope), phenotype=25 multi-label (BCE/AUROC/AUPRC)")
    ap.add_argument("--phenotype_npy", default="eicu_phenotype.npy", help="phenotype labels (N,25)")
    ap.add_argument("--no_vital_summary", action="store_true",
                    help="disable adding the vital summary statistics as side features (ablation; same meaning as central)")
    ap.add_argument("--no_static", action="store_true",
                    help="disable adding the static clinical features (eicu_static.npy) as side features (ablation)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gamma", type=float, nargs=3, default=[0.3, 0.3, 0.2])
    ap.add_argument("--lambda1", type=float, default=0.2)
    ap.add_argument("--lambda2", type=float, default=0.2)
    ap.add_argument("--lambda3", type=float, default=0.2)
    ap.add_argument("--top_m", type=int, nargs="+", default=[15, 30, 45],
                    help="per-tier top-M [client edge region]; T4=everything else. a single int is also allowed")
    # entropy thresholds (72-class max entropy = ln72 ~= 4.28; sweeping via a post-hoc trade-off curve is recommended)
    ap.add_argument("--e1", type=float, default=1.3)
    ap.add_argument("--e2", type=float, default=2.4)
    ap.add_argument("--e3", type=float, default=3.2)
    ap.add_argument("--max_batches", type=int, default=0, help="cap on batches per round per client (0=unlimited)")
    ap.add_argument("--limit_clients", type=int, default=0, help="limit the number of clients (for debugging)")
    ap.add_argument("--eval_every", type=int, default=0, help="0 = detailed eval only every tau4")
    ap.add_argument("--class_weight", action="store_true",
                    help="inverse-frequency class weights (default OFF; may trade off against accuracy -- for flattening the per-task curve)")
    ap.add_argument("--lr_sched", choices=["none", "cosine"], default="none")
    # ---- plain single-model probe (no exits/tiers; to check data learnability, not a baseline) ----
    ap.add_argument("--single_model", action="store_true",
                    help="centralized training with a single backbone+head and no exits/tiers (learnability probe). --centralized is applied automatically")
    ap.add_argument("--single_layers", type=int, default=2,
                    help="number of backbone layers for the single model (GRU/LSTM num_layers or Transformer encoder layers)")
    ap.add_argument("--final_local_adapt", type=int, default=1,
                    help="number of per-client re-adaptation epochs after training ends (last agg) -> evaluate/deploy in the personalized (adapted) state")
    # ---- FL baseline (phenotype only): if set, run baselines_eicu.run_baseline instead of MT-SFL, then exit ----
    ap.add_argument("--baseline", default="",
                    help="run an FL baseline: hfl|ebfl|hsfl|phsfl|pfl|lgfedavg")
    ap.add_argument("--phsfl_temp_tune", type=int, default=1,
                    help="PH-SFL: number of temporary head fine-tune epochs at eval time (restored afterwards)")
    ap.add_argument("--phsfl_final_tune", type=int, default=10,
                    help="PH-SFL: number of final per-client head fine-tune epochs")
    # ---- Stage B: phenotype routed evaluation + E_th sweep (only active when enabled; leaves existing eval unchanged) ----
    ap.add_argument("--tune_eth", action="store_true",
                    help="run routed eval + E_th sweep on the eval set right after training (val if val_frac>0) (phenotype only)")
    ap.add_argument("--eth_fixed", default="",
                    help="if given 'e1,e2,e3', run routed eval once at those thresholds with no sweep (for test reporting)")
    ap.add_argument("--eth_monitor", default="",
                    help="if given 'e1,e2,e3', during periodic eval show routed/routed-LW/"
                         "per-client(LW) at these fixed thresholds (progress monitoring). The final reported E_th is chosen separately by the --tune_eth sweep.")
    ap.add_argument("--eth_tolerance", type=float, default=0.005,
                    help="among candidates with routed macro-AUROC >= cloud-only macro-AUROC - tol, pick the one with the lowest cloud usage")
    ap.add_argument("--eth_npct", type=int, default=7, help="number of entropy-percentile candidates per exit")
    ap.add_argument("--eth_p_lo", type=float, default=10.0, help="lower bound of the percentile candidates")
    ap.add_argument("--eth_p_hi", type=float, default=95.0, help="upper bound of the percentile candidates")
    ap.add_argument("--latency_sweep", action="store_true",
                    help="at each eval, print the accuracy-latency trade-off (routed-LW) operating points as [LAT-PT] logs")
    ap.add_argument("--eth_metric", choices=["samplewise","lw"], default="samplewise",
                    help="E_th selection criterion: samplewise(routed) or lw(routed-LW)")
    # ---- test-set construction mode (JSAC revision: label-shift / temporal evaluation) ----
    ap.add_argument("--test_mode", choices=["iid", "labelshift", "temporal"], default="iid",
                    help="test construction: iid=existing split / labelshift=atypical patients per client (label distribution differs from the mean) / "
                         "temporal=most-recent admissions per client. all three keep the same ratio and apply to both MT-SFL and baselines")
    ap.add_argument("--test_time_key", default="",
                    help="column name of the temporal ordering key (used if present in meta; otherwise the patientunitstayid proxy)")
    ap.add_argument("--labelshift_temp", type=float, default=0.0,
                    help="labelshift strength: 0=deterministic max shift (TVD~=1.2); larger relaxes it (8 -> ~=0.34)")
    # ---- select E_th on a shift-matched val rather than test (avoids test-set tuning) ----
    ap.add_argument("--eth_select_on", choices=["test", "val"], default="test",
                    help="E_th sweep target: test=existing (select on test) / "
                         "val=split a val from train with the same rule as test_mode -> select on val -> report on test once")
    ap.add_argument("--eth_val_frac", type=float, default=0.15,
                    help="fraction of each client's train taken as a shift-matched val when --eth_select_on val")
    args = ap.parse_args()
    if args.single_model:                                  # the single-model probe uses the pooled (centralized) data path
        args.centralized = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load data + standardize + build 8 channels ----
    X = np.load(os.path.join(args.data_dir, "eicu_X.npy"))
    Mk = np.load(os.path.join(args.data_dir, "eicu_mask.npy")).astype(np.float32)
    meta = pd.read_csv(os.path.join(args.data_dir, "eicu_meta.csv"))
    vocab = json.load(open(os.path.join(args.data_dir, "eicu_label_vocab.json")))
    n_classes = len(vocab)
    is_pheno = args.task == "phenotype"
    pheno = None
    if is_pheno:
        pheno = np.load(os.path.join(args.data_dir, args.phenotype_npy)).astype(np.float32)  # (N,25)
        assert pheno.shape[0] == X.shape[0], f"phenotype rows {pheno.shape[0]} != X {X.shape[0]}"
        n_classes = pheno.shape[1]
        print(f"[task] phenotype multi-label: {n_classes} phenotypes (BCEWithLogits / AUROC·AUPRC)")
    else:
        print(f"[task] diagnosis single-label: {n_classes} classes (CrossEntropy / acc / scope)")

    # region join
    hpath = args.hospital_csv
    if hpath is None:
        for cand in [os.path.join(args.data_dir, "hospital.csv"),
                     os.path.join(args.data_dir, "..", "hospital.csv"), "hospital.csv"]:
            if os.path.exists(cand):
                hpath = cand
                break
    hosp = pd.read_csv(hpath)
    hosp.columns = [c.strip().lower() for c in hosp.columns]
    h2r = dict(zip(hosp["hospitalid"], hosp["region"]))
    meta["region"] = meta["hospitalid"].map(h2r)

    # ---- effective split (split_eff): rebuild the test set per test_mode (fixed before standardization) ----
    # iid matches the existing split. labelshift/temporal reselect the test at the same ratio per client.
    # standardization, train/eval masks, baselines, and the E_th sweep all inherit split_eff afterwards -> fair comparison.
    from eicu_test_splits import assign_effective_split
    meta["split_eff"] = assign_effective_split(
        meta, pheno, args.test_mode, args.test_time_key,
        args.seed, args.labelshift_temp, verbose=True).to_numpy()

    is_tr = (meta["split_eff"] == "train").to_numpy()
    for c in range(4):  # standardize with train observation statistics
        obs = Mk[is_tr, :, c] == 1
        vals = X[is_tr, :, c][obs]
        mu, sd = vals.mean(), vals.std() + 1e-6
        X[:, :, c] = (X[:, :, c] - mu) / sd
    # vital summary (original 4 channels, after standardization and before extra concat -- same as central_multiexit_eicu.py)
    vsum = vital_summaries(X, Mk) if not args.no_vital_summary else None
    if args.extra_channels:                                # extra time-series channels (strengthen the backbone)
        Xe = np.load(os.path.join(args.data_dir, f"eicu_X_extra{args.extra_suffix}.npy")).astype(np.float32)
        Mke = np.load(os.path.join(args.data_dir, f"eicu_mask_extra{args.extra_suffix}.npy")).astype(np.float32)
        for c in range(Xe.shape[2]):
            obs = Mke[is_tr, :, c] == 1
            vals = Xe[is_tr, :, c][obs]
            if vals.size > 1:
                mu, sd = vals.mean(), vals.std() + 1e-6
                Xe[:, :, c] = (Xe[:, :, c] - mu) / sd
        X = np.concatenate([X, Xe], axis=2); Mk = np.concatenate([Mk, Mke], axis=2)
        print(f"extra time-series channels +{Xe.shape[2]} (vitals 4 -> {X.shape[2]})")
    feat = np.concatenate([X, Mk], axis=2).transpose(0, 2, 1).astype(np.float32)  # (N, 2C, 288)

    # temporal mean-pooling (GRU speed/stability): (N,8,288) -> (N,8,288//k)
    if args.time_pool > 1:
        T = feat.shape[2]; k = args.time_pool; Tk = (T // k) * k
        feat = feat[:, :, :Tk].reshape(feat.shape[0], feat.shape[1], Tk // k, k).mean(axis=3)
        print(f"time_pool={k}: sequence length {T} -> {feat.shape[2]}")
    n_ch = feat.shape[1]   # number of backbone input channels (= 2*(4+extra)) -- reflects extra_channels

    # ---- load/standardize the lab vector (if present): standardized [min,max,last] x K + measured-flag x K ----
    lab_path = os.path.join(args.data_dir, "eicu_labs.npy")
    if os.path.exists(lab_path):
        L = np.load(lab_path).astype(np.float32)               # (N,K,3)
        LM = np.load(os.path.join(args.data_dir, "eicu_lab_mask.npy")).astype(np.float32)  # (N,K)
        Nrow, K, _ = L.shape
        Lflat = L.reshape(Nrow, K * 3)
        mask3 = np.repeat(LM, 3, axis=1)                        # (N,3K) broadcast the lab flag across the 3 stats
        for j in range(Lflat.shape[1]):                        # standardize each column with train observations
            obs = (mask3[is_tr, j] == 1)
            v = Lflat[is_tr, j][obs]
            if v.size > 1:
                mu, sd = v.mean(), v.std() + 1e-6
                Lflat[:, j] = (Lflat[:, j] - mu) / sd
        Lflat[mask3 == 0] = 0.0                                # not measured -> 0 (mean), distinguished by the flag
        labvec = np.concatenate([Lflat, LM], axis=1).astype(np.float32)  # (N, 3K+K)
        lab_dim = labvec.shape[1]
        print(f"lab fusion: K={K} labs, lab_dim={lab_dim}")
    else:
        labvec = np.zeros((feat.shape[0], 0), dtype=np.float32)
        lab_dim = 0
        print("no labs (vital-only)")

    # ---- add the vital summary + static clinical features to side (same as central_multiexit_eicu.py) ----
    if vsum is not None:
        labvec = np.concatenate([labvec, vsum], axis=1).astype(np.float32)
        lab_dim = labvec.shape[1]
        print(f"added vital summary: +{vsum.shape[1]} dims -> side_dim={lab_dim}")
    sp = os.path.join(args.data_dir, "eicu_static.npy")
    if not args.no_static and os.path.exists(sp):
        st = np.load(sp).astype(np.float32)
        assert st.shape[0] == labvec.shape[0], f"static row count mismatch {st.shape[0]} vs {labvec.shape[0]}"
        labvec = np.concatenate([labvec, st], axis=1).astype(np.float32)
        lab_dim = labvec.shape[1]
        print(f"added static clinical features: +{st.shape[1]} dims (apache/demographics/respiratory/comorbidity/medication) -> side_dim={lab_dim}")
    elif not args.no_static:
        print("[warn] eicu_static.npy missing -> skipping static features (run build_eicu_static_features.py first)")

    # ---- build the hierarchy index ----
    clients_list = sorted(meta["client"].unique())
    if args.limit_clients:
        clients_list = clients_list[:args.limit_clients]
        meta = meta[meta["client"].isin(clients_list)].copy()
    cl2idx = {c: i for i, c in enumerate(clients_list)}
    meta["client_idx"] = meta["client"].map(cl2idx)
    n_clients = len(clients_list)

    # client -> hospital, region
    cm = meta.drop_duplicates("client_idx").set_index("client_idx")
    hosp_of = cm["hospitalid"].to_dict()
    region_of = cm["region"].to_dict()
    hosp2clients = defaultdict(list)
    region2clients = defaultdict(list)
    region2hosps = defaultdict(set)
    for c in range(n_clients):
        hosp2clients[hosp_of[c]].append(c)
        region2clients[region_of[c]].append(c)
        region2hosps[region_of[c]].add(hosp_of[c])
    region2hosps = {r: sorted(hs) for r, hs in region2hosps.items()}
    all_idx = list(range(n_clients))

    # hierarchy maps for aggregation: members_by[l]=tier-l node->descendant clients, anc[l]=client->tier-l node
    members_by = {2: dict(hosp2clients), 3: dict(region2clients), 4: {"_global": all_idx}}
    anc = {2: {c: hosp_of[c] for c in range(n_clients)},
           3: {c: region_of[c] for c in range(n_clients)},
           4: {c: "_global" for c in range(n_clients)}}

    print(f"device={device}  clients={n_clients}  hospitals={len(hosp2clients)}  "
          f"regions={len(region2hosps)}  classes={n_classes}")

    # ---- train/val split (if val_frac>0, split a tuning val out of train at patient level) ----
    split_arr = meta["split_eff"].to_numpy()
    is_train = split_arr == "train"
    if args.val_frac > 0:
        pats = meta.loc[is_train, "patienthealthsystemstayid"].unique()
        rng = np.random.default_rng(args.seed)
        n_val = max(1, int(len(pats) * args.val_frac))
        val_pat = set(rng.choice(pats, size=n_val, replace=False))
        is_val = is_train & meta["patienthealthsystemstayid"].isin(val_pat).to_numpy()
        train_mask = is_train & ~is_val
        eval_mask = is_val
        print(f"val_frac={args.val_frac}: train {int(train_mask.sum()):,} / "
              f"val {int(eval_mask.sum()):,} (tuning-only val, test not used)")
    else:
        train_mask = is_train
        eval_mask = split_arr == "test"

    # ---- val dedicated to E_th selection: split from train at patient level to avoid test-set threshold tuning.
    # in label-shift/temporal experiments, tuning the operating point (E_th) to the test distribution weakens
    # the robustness claim, so with --eth_select_on val we pick E_th on the val split here and report on test once.
    # (if --eth_fixed is given there is no sweep, so no split is made) ----
    ethval_mask = np.zeros(len(meta), dtype=bool)
    if args.tune_eth and args.eth_select_on == "val" and not args.eth_fixed.strip():
        tr_pats = meta.loc[train_mask, "patienthealthsystemstayid"].unique()
        rng2 = np.random.default_rng(args.seed + 12345)
        n_ev = max(1, int(len(tr_pats) * args.eth_val_frac))
        ev_pat = set(rng2.choice(tr_pats, size=n_ev, replace=False))
        ethval_mask = train_mask & meta["patienthealthsystemstayid"].isin(ev_pat).to_numpy()
        train_mask = train_mask & ~ethval_mask
        print(f"[eth_select_on=val] holding out {int(ethval_mask.sum()):,} val samples from train for E_th selection "
              f"(excluded from training, val_frac={args.eth_val_frac}); the final routed metric is reported once on test")

    # ---- scope labels (T_{k,l}): derived only from the training data (train_mask) ----
    meta_tr = meta[train_mask]
    scope = derive_scope_labels(meta_tr, n_clients, cl2idx, hosp_of, region_of,
                                hosp2clients, region2clients, args.top_m)
    avg_sizes = {k: np.mean([len(scope[c][k]) for c in range(n_clients)])
                 for k in ["client", "edge", "region", "outside"]}
    print("avg |scope| (T1/T2/T3/T4, outside=T4): "
          + " / ".join(f"{avg_sizes[k]:.1f}" for k in ["client", "edge", "region", "outside"]))

    # ---- per-client DataLoaders (tensor slices) ----
    feat_t = torch.tensor(feat)
    lab_t = torch.tensor(labvec)
    # diagnosis: single-label int -> torch long (CE). phenotype: (N,25) float -> torch float (BCE).
    yall = pheno if is_pheno else meta["y"].to_numpy()
    idx_all = meta["client_idx"].to_numpy()
    row = np.arange(len(meta))
    train_loaders, test_loaders, ethval_loaders = {}, {}, {}
    counts = {}                                          # |D_k| = number of train samples per client
    for c in range(n_clients):
        tr_rows = row[(idx_all == c) & train_mask]
        ev_rows = row[(idx_all == c) & eval_mask]
        xv_rows = row[(idx_all == c) & ethval_mask]      # val for E_th selection (usually empty)
        counts[c] = len(tr_rows)
        gidx_tr = meta.index.to_numpy()[tr_rows]   # original feat/lab indices
        gidx_ev = meta.index.to_numpy()[ev_rows]
        gidx_xv = meta.index.to_numpy()[xv_rows]
        if len(gidx_tr):
            train_loaders[c] = DataLoader(
                TensorDataset(feat_t[gidx_tr], lab_t[gidx_tr],
                              torch.tensor(yall[tr_rows])),
                batch_size=args.batch_size, shuffle=True)
        test_loaders[c] = (feat_t[gidx_ev], lab_t[gidx_ev], torch.tensor(yall[ev_rows]))
        ethval_loaders[c] = (feat_t[gidx_xv], lab_t[gidx_xv], torch.tensor(yall[xv_rows]))
    n_cold = sum(1 for c in range(n_clients) if counts[c] == 0)
    if n_cold:
        print(f"cold clients with 0 train samples: {n_cold} (excluded from the aggregation average; receive only)")

    # ---- model / optimizer ----
    if args.centralized:
        # centralized for comparison: all clients point to one shared model (single copy in memory) -> per-client eval reused as-is
        if args.single_model:                              # plain single model with no exits/tiers (learnability probe)
            central = SingleEICUModel(n_ch, args.hidden[-1], lab_dim, n_classes, args.rnn,
                                      args.dropout, args.single_layers, args.nhead, args.ff_mult,
                                      multilabel=is_pheno).to(device)
        else:
            central = FullClientModel(n_classes, device, lab_dim, args.dropout,
                                      tuple(args.hidden), args.rnn, args.bidirectional, n_ch=n_ch,
                                      multilabel=is_pheno, nhead=args.nhead, ff_mult=args.ff_mult)
        if args.optimizer == "adam":
            central_opt = torch.optim.Adam(central.parameters(), lr=args.lr,
                                           weight_decay=args.weight_decay)
        else:
            central_opt = torch.optim.SGD(central.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
        clients = [central for _ in range(n_clients)]   # same object reference -> only one model loaded on the GPU
        optims = [central_opt]
        # pooled train loader: all train rows as a single dataset
        all_tr_rows = row[train_mask]
        gidx_all = meta.index.to_numpy()[all_tr_rows]
        central_loader = DataLoader(
            TensorDataset(feat_t[gidx_all], lab_t[gidx_all], torch.tensor(yall[all_tr_rows])),
            batch_size=args.batch_size, shuffle=True)
        print(f"[central] pooled train rows={len(all_tr_rows):,} -> train as a single shared model (no aggregation)")
    else:
        clients = []
        optims = []
        for c in range(n_clients):
            fm = FullClientModel(n_classes, device, lab_dim, args.dropout,
                                 tuple(args.hidden), args.rnn, args.bidirectional, n_ch=n_ch,
                                 multilabel=is_pheno, nhead=args.nhead, ff_mult=args.ff_mult)
            clients.append(fm)
            if args.optimizer == "adam":
                optims.append(torch.optim.Adam(fm.parameters(), lr=args.lr,
                                               weight_decay=args.weight_decay))
            else:
                optims.append(torch.optim.SGD(fm.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay))

    if args.single_model:                                  # ===== single-model learnability probe with no exits =====
        npar = sum(p.numel() for p in central.parameters())
        print(f"\n[single-model probe] plain {args.rnn} single model with no exits/tiers  "
              f"layers={args.single_layers} hidden={args.hidden[-1]} params={npar:,}  (task={args.task})", flush=True)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(central_opt, T_max=max(1, args.num_epochs))
                 if args.lr_sched == "cosine" else None)

        def _single_eval(tag):
            central.eval()
            P, Tl = [], []
            for c in range(n_clients):
                xb, lb, yb = test_loaders[c]
                if len(yb) == 0:
                    continue
                with torch.no_grad():
                    for s in range(0, len(yb), 512):
                        lg = central(xb[s:s + 512].to(device), lb[s:s + 512].to(device))
                        P.append((torch.sigmoid(lg) if is_pheno else lg).cpu().numpy())
                        Tl.append(yb[s:s + 512].numpy())
            Tall = np.concatenate(Tl, 0); Pall = np.concatenate(P, 0)
            if is_pheno:
                au, ap = [], []
                for k in range(n_classes):
                    yk = Tall[:, k]
                    if yk.min() == yk.max():
                        continue
                    au.append(roc_auc_score(yk, Pall[:, k])); ap.append(average_precision_score(yk, Pall[:, k]))
                r = {"macro_auroc": float(np.mean(au)) if au else 0.0,
                     "macro_auprc": float(np.mean(ap)) if ap else 0.0, "n": int(len(Tall)), "n_pheno": len(au)}
                print(f"  [{tag}] macro-AUROC={r['macro_auroc']:.4f}  macro-AUPRC={r['macro_auprc']:.4f}  "
                      f"(pooled {r['n']} samples, {r['n_pheno']} phenotypes)", flush=True)
            else:
                pred = Pall.argmax(1); r = {"acc": float((pred == Tall).mean()), "n": int(len(Tall))}
                print(f"  [{tag}] acc={r['acc']:.4f}  (pooled {r['n']} samples)", flush=True)
            return r

        for epoch in range(args.num_epochs):
            central.train()
            for x, lab, y in central_loader:
                x, lab, y = x.to(device), lab.to(device), y.to(device)
                lg = central(x, lab)
                loss = (F.binary_cross_entropy_with_logits(lg, y) if is_pheno else F.nll_loss(lg, y))
                central_opt.zero_grad(); loss.backward(); central_opt.step()
            if sched is not None:
                sched.step()
            if args.eval_every and (epoch + 1) % args.eval_every == 0:
                _single_eval(f"epoch {epoch+1}")
        res = _single_eval("final")
        out = os.path.join(args.data_dir, f"{args.run_tag or 'single'}_probe.json")
        json.dump({"run_tag": args.run_tag, "task": args.task, "rnn": args.rnn,
                   "layers": args.single_layers, "hidden": args.hidden[-1], "params": npar,
                   "result": res}, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"\n[single-model probe done] -> {out}", flush=True)
        return

    def nparams(m):
        return sum(p.numel() for p in m.parameters())
    f0 = clients[0]
    print(f"per-tier params (client→cloud): "
          f"{nparams(f0.client):,} / {nparams(f0.edge):,} / "
          f"{nparams(f0.region):,} / {nparams(f0.cloud):,}  "
          f"(rnn={args.rnn}, hidden={args.hidden}, bidir={args.bidirectional})")

    # class weights (optional, default OFF) -- inverse-frequency of train labels. not applied to phenotype (multi-label).
    class_weight = None
    if args.class_weight and not is_pheno:
        freq = np.ones(n_classes, dtype=np.float64)
        for lab_id in yall[train_mask]:
            if 0 <= int(lab_id) < n_classes:
                freq[int(lab_id)] += 1
        w = (1.0 / freq); w = w * (n_classes / w.sum())
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("class weighting = ON (inverse-freq)")

    # LR scheduler (optional)
    scheds = []
    if args.lr_sched == "cosine":
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs))
                  for o in optims]
        print(f"LR schedule = cosine over {args.num_epochs} epochs")

    def evaluate(detailed):
        # per-client (macro-over-clients) accuracy. average per-client acc without pooling the test set.
        tier_names = ["client", "edge", "region", "cloud"]
        scope_names = ["client", "edge", "region", "outside"]   # outside = T4
        label_map = {"client": "T1", "edge": "T2", "region": "T3", "outside": "T4"}
        exit_cnt = Counter()
        per_overall = []
        per_scope = {s: [] for s in scope_names}
        per_client_metric = []                              # scope=client(T1) per-client acc
        all_yn, all_pn = [], []                             # for pooled macro-F1
        for c in range(n_clients):
            xb, lb, yb = test_loaders[c]
            if len(yb) == 0:
                continue
            sc = scope[c]
            clients[c].eval_mode()
            n = ok = 0
            scn = {s: 0 for s in scope_names}
            scc = {s: 0 for s in scope_names}
            with torch.no_grad():
                for s in range(0, len(yb), 512):
                    xx = xb[s:s + 512].to(device)
                    ll = lb[s:s + 512].to(device)
                    yy = yb[s:s + 512].to(device)
                    preds, tiers = infer_with_exits(clients[c], xx, ll, args.e1, args.e2, args.e3)
                    pn = preds.cpu().numpy(); yn = yy.cpu().numpy(); tn = tiers.cpu().numpy()
                    all_yn.append(yn); all_pn.append(pn)
                    for j in range(len(yn)):
                        exit_cnt[tier_names[int(tn[j])]] += 1
                        lbl = int(yn[j]); good = int(pn[j] == lbl)
                        n += 1; ok += good
                        st = ("client" if lbl in sc["client"] else "edge" if lbl in sc["edge"]
                              else "region" if lbl in sc["region"] else "outside")
                        scn[st] += 1; scc[st] += good
            if n == 0:
                continue
            per_overall.append(ok / n)
            for s in scope_names:
                if scn[s] > 0:
                    per_scope[s].append(scc[s] / scn[s])
            if scn["client"] > 0:
                per_client_metric.append(scc["client"] / scn["client"])
        acc = float(np.mean(per_overall)) if per_overall else 0.0
        tot = sum(exit_cnt.values())
        mf1 = macro_f1(np.concatenate(all_yn), np.concatenate(all_pn), n_classes) if all_yn else 0.0
        print(f"  per-client acc {acc:.3f}  macro-F1 {mf1:.4f}  (clients={len(per_overall)})  | exit ratios: " +
              " ".join(f"{t} {exit_cnt[t]/max(tot,1):.0%}" for t in tier_names))
        if detailed:
            print("  per-client acc by scope (T1=client ... T4=outside):")
            for s in scope_names:
                if per_scope[s]:
                    print(f"    {label_map[s]:3s}: {np.mean(per_scope[s]):.2%} (clients={len(per_scope[s])})")
                else:
                    print(f"    {label_map[s]:3s}: n/a")
        client_acc = float(np.mean(per_client_metric)) if per_client_metric else 0.0
        return {"task": "diagnosis", "acc": acc, "client_acc": client_acc, "macro_f1": mf1,
                "n_clients": len(per_overall),
                "scope": {label_map[s]: (float(np.mean(per_scope[s])) if per_scope[s] else None) for s in scope_names},
                "exit_ratio": {t: (exit_cnt[t] / tot if tot else 0.0) for t in tier_names}}

    def evaluate_pheno(detailed):
        # multi-label phenotype: evaluate each exit on all test samples instead of early-exit routing/scope
        # -> compute per-phenotype AUROC/AUPRC pooled (over clients/samples), then macro-average.
        # (early-exit reallocates for compute savings, not accuracy routing -- per-exit performance is over all samples)
        names = ["client", "edge", "region", "cloud"]
        is_baseline = bool(getattr(args, "baseline", "").strip())   # baselines report the full model only
        # if fixed E_th (for reporting) or eth_monitor (for progress) is set, also show routed/routed-LW in periodic eval.
        mon = args.eth_fixed.strip() or args.eth_monitor.strip()
        want_routed = bool(mon)
        P = [[] for _ in range(4)]; T = []; Hl = []; client_sizes = []
        for c in range(n_clients):
            xb, lb, yb = test_loaders[c]
            if len(yb) == 0:
                continue
            client_sizes.append(int(len(yb)))
            full = clients[c]; full.eval_mode()
            with torch.no_grad():
                for s in range(0, len(yb), 512):
                    xx = xb[s:s + 512].to(device)
                    ll = lb[s:s + 512].to(device)
                    c_out, c_aux = full.client(xx, ll)
                    e_out, e_aux = full.edge(c_out, ll)
                    r_out, r_aux = full.region(e_out, ll)
                    cl_out = full.cloud(r_out, ll)              # multilabel -> raw logits
                    for m, lg in enumerate((c_aux, e_aux, r_aux, cl_out)):
                        P[m].append(torch.sigmoid(lg).cpu().numpy())
                    if want_routed:
                        Hl.append(torch.stack([mean_binary_entropy(c_aux), mean_binary_entropy(e_aux),
                                               mean_binary_entropy(r_aux)], dim=1).cpu().numpy())
                    T.append(yb[s:s + 512].numpy())
        if not T:
            return 0.0, 0.0, 0.0
        Tall = np.concatenate(T, 0)
        macro = []
        for m in range(4):
            Pm = np.concatenate(P[m], 0); au = []; ap = []
            for k in range(n_classes):
                yk = Tall[:, k]
                if yk.min() == yk.max():            # single class -> skip in macro
                    continue
                au.append(roc_auc_score(yk, Pm[:, k])); ap.append(average_precision_score(yk, Pm[:, k]))
            yf, pf = Tall.ravel(), Pm.ravel()       # micro: single pooled computation over all (sample x phenotype)
            ok = yf.min() != yf.max()
            macro.append({"macro_auroc": float(np.mean(au)) if au else 0.0,
                          "micro_auroc": float(roc_auc_score(yf, pf)) if ok else 0.0,
                          "macro_auprc": float(np.mean(ap)) if ap else 0.0,
                          "micro_auprc": float(average_precision_score(yf, pf)) if ok else 0.0})
        if not is_baseline:
            print("  per-exit macro/micro-AUROC: " +
                  " ".join(f"{names[m]}({macro[m]['macro_auroc']:.4f}/{macro[m]['micro_auroc']:.4f})" for m in range(4)) +
                  f"  (pooled {len(Tall)} samples)")
        if detailed and not is_baseline:
            print("  per-exit macro/micro-AUPRC: " +
                  " ".join(f"{names[m]}({macro[m]['macro_auprc']:.4f}/{macro[m]['micro_auprc']:.4f})" for m in range(4)))
        routed_info = None
        lw_routed = None
        if want_routed:                                     # ---- fixed/monitor E_th routed = early-exit operating metric ----
            e1, e2, e3 = (float(v) for v in mon.split(","))
            H = np.concatenate(Hl, 0)
            tier = _route_tiers(H[:, 0], H[:, 1], H[:, 2], e1, e2, e3)
            Pall = np.stack([np.concatenate(P[m], 0) for m in range(4)], axis=0)   # (4,N,C)
            routed_P = Pall[tier, np.arange(len(tier)), :]                          # per-sample routed exit probabilities
            r_au, r_mi, r_ap, _ = _macro_micro(Tall, routed_P, n_classes)
            ratio = np.bincount(tier, minlength=4) / len(tier)
            src_tag = "fixed" if args.eth_fixed.strip() else "monitor"
            print(f"  [routed {src_tag} E_th=({e1:.3f},{e2:.3f},{e3:.3f})] macro-AUROC={r_au:.4f} "
                  f"micro={r_mi:.4f} macro-AUPRC={r_ap:.4f}  "
                  f"exit%: cl={ratio[0]:.0%} ed={ratio[1]:.0%} rg={ratio[2]:.0%} cd={ratio[3]:.0%}")
            routed_info = {"routed_macro_auroc": r_au, "routed_micro_auroc": r_mi,
                           "routed_macro_auprc": r_ap, "exit_ratio": [float(x) for x in ratio]}
            # ---- label-wise routed: an independent exit per (sample,label). broadcast the same E_th over the 25 labels ----
            lw_mm, lw_lr, lw_dr, lw_routed = _labelwise_routed(Pall, Tall, e1, e2, e3, n_classes)
            print(f"  [routed-LW {src_tag} E_th=({e1:.3f},{e2:.3f},{e3:.3f})] macro-AUROC={lw_mm[0]:.4f} "
                  f"micro={lw_mm[1]:.4f} macro-AUPRC={lw_mm[2]:.4f}  "
                  f"label-exit%: cl={lw_lr[0]:.0%} ed={lw_lr[1]:.0%} rg={lw_lr[2]:.0%} cd={lw_lr[3]:.0%} | "
                  f"sample-depth%: cl={lw_dr[0]:.0%} ed={lw_dr[1]:.0%} rg={lw_dr[2]:.0%} cd={lw_dr[3]:.0%}")
            routed_info.update({
                "labelwise_macro_auroc": lw_mm[0], "labelwise_micro_auroc": lw_mm[1],
                "labelwise_macro_auprc": lw_mm[2], "labelwise_micro_auprc": lw_mm[3],
                "labelwise_label_exit_ratio": [float(x) for x in lw_lr],     # label-level
                "labelwise_sample_depth_ratio": [float(x) for x in lw_dr],   # sample-level (propagation depth)
                "labelwise_cloud_usage": float(lw_dr[3])})
            if getattr(args, "latency_sweep", False):
                # ---- accuracy-latency trade-off for Fig. 6 (routed-LW). latency = sample_depth-weighted per Eqs. 34-36 ----
                print("  [LAT-SWEEP] accuracy-latency operating points (routed-LW); latency=f(sample_depth) per paper")
                for _m, _nm in enumerate(["client", "edge", "region", "cloud"]):
                    _sd = [0.0, 0.0, 0.0, 0.0]; _sd[_m] = 1.0
                    print(f"    [LAT-PT] op=exit_{_nm} macroAUROC={macro[_m]['macro_auroc']:.4f} "
                          f"sample_depth=[{_sd[0]:.4f},{_sd[1]:.4f},{_sd[2]:.4f},{_sd[3]:.4f}] "
                          f"label_exit=[{_sd[0]:.4f},{_sd[1]:.4f},{_sd[2]:.4f},{_sd[3]:.4f}]")
                for _a in [0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00]:
                    _se1, _se2, _se3 = e1*_a, e2*_a, e3*_a
                    _mm, _lr, _dr, _ = _labelwise_routed(Pall, Tall, _se1, _se2, _se3, n_classes)
                    _mk = " *op" if abs(_a-1.0) < 1e-9 else ""
                    print(f"    [LAT-PT] op=routedLW_a{_a:.2f} eth=({_se1:.4f},{_se2:.4f},{_se3:.4f}) "
                          f"macroAUROC={_mm[0]:.4f} "
                          f"sample_depth=[{_dr[0]:.4f},{_dr[1]:.4f},{_dr[2]:.4f},{_dr[3]:.4f}] "
                          f"label_exit=[{_lr[0]:.4f},{_lr[1]:.4f},{_lr[2]:.4f},{_lr[3]:.4f}]{_mk}")
        # ---- per-client macro-AUROC (macro-over-clients): MT-SFL uses routed-LW, baselines use the cloud prediction ----
        P_pc = lw_routed if want_routed else np.concatenate(P[3], 0)         # per-client from routed-LW (N,C) probabilities
        pc_au, pc_n, pc_lab = _per_client_macro_auroc(Tall, P_pc, client_sizes)
        pc_src = "routed-LW" if want_routed else "full model"
        if not is_baseline:
            print(f"  [per-client macro-AUROC ({pc_src})] {pc_au:.4f}  "
                  f"(over {pc_n}/{len(client_sizes)} clients, avg {pc_lab:.1f} valid labels/client)")
        cloud_auroc = macro[3]["macro_auroc"]
        names_t = ["client", "edge", "region", "cloud"]
        return {"task": "phenotype", "acc": cloud_auroc, "client_acc": cloud_auroc,
                "cloud_macro_auroc": cloud_auroc, "routed": routed_info,
                "per_client_macro_auroc": pc_au, "per_client_src": pc_src, "per_client_n": pc_n,
                "per_exit": {names_t[m]: macro[m] for m in range(4)}}

    # ---- main loop ----
    best = {"acc": -1.0, "client_acc": 0.0, "epoch": 0}

    # ===== FL baseline branch (if set, run the baseline instead of the MT-SFL loop, then exit) =====
    if args.baseline:
        from baselines_eicu import run_baseline
        ctx = dict(clients=clients, optims=optims, train_loaders=train_loaders,
                   test_loaders=test_loaders, counts=counts, scope=scope,
                   members_by=members_by, anc=anc, n_clients=n_clients,
                   n_classes=n_classes, lab_dim=lab_dim, device=device,
                   is_pheno=is_pheno, args=args,
                   evaluate=evaluate, evaluate_pheno=evaluate_pheno)
        run_baseline(args.baseline, ctx)
        return

    summary_rows = []

    def track(ep):
        m = evaluate_pheno(detailed=True) if is_pheno else evaluate(detailed=True)
        a = m["acc"]; ca = m["client_acc"]
        if a > best["acc"]:
            best.update(acc=a, client_acc=ca, epoch=ep)
        m["epoch"] = ep
        summary_rows.append(m); write_summary_eicu(args, summary_rows)
        return m

    for ep in range(1, args.num_epochs + 1):
        if args.centralized:
            for _ in range(args.local_epochs):
                client_train(central, central_loader, central_opt, args.gamma,
                             device, args.max_batches, class_weight, multilabel=is_pheno)
            for s in scheds:
                s.step()
        else:
            for c in range(n_clients):
                if c in train_loaders:
                    for _ in range(args.local_epochs):
                        client_train(clients[c], train_loaders[c], optims[c], args.gamma,
                                     device, args.max_batches, class_weight, multilabel=is_pheno)
            for s in scheds:
                s.step()
            # per-tier period E_m=(tau2,tau3,tau4). only the highest triggering tier cascades (including lower tiers).
            if not args.no_agg:
                m_star = 0
                if ep % args.tau4 == 0:
                    m_star = 4
                elif ep % args.tau3 == 0:
                    m_star = 3
                elif ep % args.tau2 == 0:
                    m_star = 2
                if m_star >= 2:
                    aggregate(clients, m_star, members_by, anc, counts,
                              [args.lambda1, args.lambda2, args.lambda3])
        if ep % args.tau4 == 0:
            tag = "centralized" if args.centralized else ("local-only" if args.no_agg else "after tier aggregation")
            print(f"[epoch {ep}] {tag} detailed eval:")
            track(ep)
        elif args.eval_every and ep % args.eval_every == 0:
            print(f"[epoch {ep}]")
            track(ep)

    print("\n=== final eval ===")
    # per-client re-adaptation after the last aggregation -> evaluate/deploy in the personalized (adapted) state (drop the full tier-4 sync).
    if not args.no_agg and not args.centralized and args.final_local_adapt > 0:
        for c in range(n_clients):
            if c in train_loaders:
                for _ in range(args.final_local_adapt):
                    client_train(clients[c], train_loaders[c], optims[c], args.gamma,
                                 device, args.max_batches, class_weight, multilabel=is_pheno)
    acc = track(args.num_epochs)["acc"]        # record best/JSON in the final state
    if args.val_frac > 0:
        result = {"run_tag": args.run_tag,
                  "best_val_acc": best["acc"], "best_val_client_acc": best["client_acc"],
                  "best_epoch": best["epoch"],
                  "config": {"lambda1": args.lambda1, "lambda2": args.lambda2,
                             "lambda3": args.lambda3, "gamma": args.gamma, "lr": args.lr,
                             "tau": [args.tau2, args.tau3, args.tau4],
                             "local_epochs": args.local_epochs, "top_m": args.top_m,
                             "dropout": args.dropout, "weight_decay": args.weight_decay}}
        out = os.path.join(args.data_dir, f"tune_{args.run_tag}.json")
        json.dump(result, open(out, "w"), indent=2)
        print(f"[tune] {out}  best_val: acc={best['acc']:.4f} "
              f"client={best['client_acc']:.4f} @ep{best['epoch']}")
    else:
        base = args.run_tag or (("central" if args.centralized else "mtsfl") + "_" + args.task)
        out_name = f"{base}_result.json"
        json.dump({"acc": float(acc), "n_clients": n_clients},
                  open(os.path.join(args.data_dir, out_name), "w"), indent=2)
    if args.tune_eth:
        if is_pheno:
            # if eth_select_on=val (and not fixed E_th), select on the val split from train and report on test.
            # otherwise pass sel==report==test_loaders to keep the existing behavior.
            use_val = (args.eth_select_on == "val" and not args.eth_fixed.strip()
                       and any(len(ethval_loaders[c][2]) for c in range(n_clients)))
            sel_loaders = ethval_loaders if use_val else test_loaders
            eth_routed_sweep(args, clients, sel_loaders, test_loaders,
                             n_clients, n_classes, device)
        else:
            print("[tune_eth] phenotype-only feature -> skipped for diagnosis")
    print("training complete.")


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def write_summary_eicu(args, rows):
    """Save all per-epoch metrics as CSV+JSON. The diagnosis/phenotype column sets are unioned automatically."""
    tag = args.run_tag or (("central" if args.centralized else "mtsfl") + "_" + args.task)
    base = os.path.join(args.data_dir, f"{tag}_summary")
    json.dump(rows, open(base + ".json", "w"), indent=2, ensure_ascii=False)
    flat = []
    for r in rows:
        if r.get("task") == "phenotype":
            d = {"epoch": r.get("epoch"), "cloud_macro_auroc": r.get("cloud_macro_auroc"),
                 "per_client_macro_auroc": r.get("per_client_macro_auroc"),
                 "per_client_src": r.get("per_client_src")}
            ri = r.get("routed") or {}                      # routed/routed-LW (when eth_monitor/fixed is set)
            d["routed_macro_auroc"] = ri.get("routed_macro_auroc")
            d["labelwise_macro_auroc"] = ri.get("labelwise_macro_auroc")
            d["labelwise_cloud_usage"] = ri.get("labelwise_cloud_usage")
            for t in ("client", "edge", "region", "cloud"):
                pe = (r.get("per_exit") or {}).get(t, {})
                d[f"{t}_macro_auroc"] = pe.get("macro_auroc"); d[f"{t}_micro_auroc"] = pe.get("micro_auroc")
                d[f"{t}_macro_auprc"] = pe.get("macro_auprc"); d[f"{t}_micro_auprc"] = pe.get("micro_auprc")
        else:
            d = {"epoch": r.get("epoch"), "acc": r.get("acc"), "macro_f1": r.get("macro_f1"),
                 "client_acc": r.get("client_acc")}
            for g in ("T1", "T2", "T3", "T4"):
                d[f"scope_{g}"] = (r.get("scope") or {}).get(g)
            for t in ("client", "edge", "region", "cloud"):
                d[f"exit_{t}"] = (r.get("exit_ratio") or {}).get(t)
        flat.append(d)
    if flat:
        keys = []
        for d in flat:
            for k in d:
                if k not in keys: keys.append(k)
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(flat)


def _macro_micro(Y, P, n_classes):
    """Y,(N,C) labels / P,(N,C) probabilities -> macro/micro AUROC/AUPRC."""
    au, ap = [], []
    for k in range(n_classes):
        yk = Y[:, k]
        if yk.min() == yk.max():
            continue
        au.append(roc_auc_score(yk, P[:, k]))
        ap.append(average_precision_score(yk, P[:, k]))
    yf, pf = Y.ravel(), P.ravel()
    ok = yf.min() != yf.max()
    return (float(np.mean(au)) if au else 0.0,
            float(roc_auc_score(yf, pf)) if ok else 0.0,
            float(np.mean(ap)) if ap else 0.0,
            float(average_precision_score(yf, pf)) if ok else 0.0)


def _per_client_macro_auroc(Y, P, client_sizes):
    """per-client (macro-over-clients) macro-AUROC.
    Y,(N,C) labels / P,(N,C) probabilities / client_sizes = number of samples per client (rows must be stacked in client order).
    For each client, per-label AUROC on its own samples -> macro within the client -> average over clients.
    A label that is single-class within a client (all 0 or all 1) is skipped for that client,
    and a client with no valid label at all (e.g. a data-poor client) is excluded from the average.
    Returns: (mean_auroc, n_valid_clients, avg_valid_labels_per_client)."""
    offs = np.cumsum([0] + [int(s) for s in client_sizes])
    per_client, n_lab = [], []
    for i in range(len(client_sizes)):
        a, b = offs[i], offs[i + 1]
        yc, pc = Y[a:b], P[a:b]
        aus = []
        for k in range(yc.shape[1]):
            col = yc[:, k]
            if col.min() == col.max():          # this label is single-class for this client -> skip
                continue
            aus.append(roc_auc_score(col, pc[:, k]))
        if aus:
            per_client.append(float(np.mean(aus))); n_lab.append(len(aus))
    if not per_client:
        return 0.0, 0, 0.0
    return float(np.mean(per_client)), len(per_client), float(np.mean(n_lab))


def _route_tiers(H1, H2, H3, e1, e2, e3):
    """sample-wise routing: H1<e1->client(0); else H2<e2->edge(1); else H3<e3->region(2); otherwise cloud(3)."""
    N = H1.shape[0]
    tier = np.full(N, 3, dtype=np.int64)
    m1 = H1 < e1
    tier[m1] = 0
    m2 = (~m1) & (H2 < e2)
    tier[m2] = 1
    m3 = (~m1) & (~m2) & (H3 < e3)
    tier[m3] = 2
    return tier


# ======================================================================================
# label-wise early-exit (phenotype 25-label only): for a single sample, the 25 labels
# each pick an exit independently. With no extra forward pass, using the per-exit probabilities
# (4,N,C) already collected by evaluate_pheno/eth_routed_sweep, routing derives a per-label binary entropy.
# ======================================================================================
def _bin_entropy_prob(P):
    """probabilities P,(...,C) -> per-label binary entropy (same shape). P is assumed to be sigmoid(logits) already.
    The same definition as mean_binary_entropy(logits), 'before averaging' (no mean(dim=1))."""
    return -(P * np.log(P + 1e-10) + (1.0 - P) * np.log(1.0 - P + 1e-10))


def _route_tiers_labelwise(H1, H2, H3, e1, e2, e3):
    """label-wise routing: an independent exit per (sample,label). H1/H2/H3,(N,C) are per-label binary entropy.
    e1/e2/e3 are scalar or (C,) -> numpy broadcast. Returns tier,(N,C):
    H1<e1->client(0); else H2<e2->edge(1); else H3<e3->region(2); otherwise cloud(3)."""
    tier = np.full(H1.shape, 3, dtype=np.int64)
    m1 = H1 < e1
    tier[m1] = 0
    m2 = (~m1) & (H2 < e2)
    tier[m2] = 1
    m3 = (~m1) & (~m2) & (H3 < e3)
    tier[m3] = 2
    return tier


def _labelwise_routed(Pstack, Y, e1, e2, e3, n_classes):
    """label-wise routed probabilities + two kinds of ratios.
    Pstack,(4,N,C) per-exit probabilities (0=client..3=cloud), Y,(N,C) labels. Returns:
      mm          = (macroAUROC, microAUROC, macroAUPRC, microAUPRC)  based on the routed probabilities
      label_ratio,(4,) = fraction of all (sample x label) predictions read at each tier  (= label-level exit ratio)
      depth_ratio,(4,) = distribution of each sample's 'deepest label tier'  (= sample-level: the depth that
                         the sample actually has to be forwarded to. communication/compute savings should be judged by this metric)
    """
    H1 = _bin_entropy_prob(Pstack[0])
    H2 = _bin_entropy_prob(Pstack[1])
    H3 = _bin_entropy_prob(Pstack[2])
    tier = _route_tiers_labelwise(H1, H2, H3, e1, e2, e3)                 # (N,C)
    routed = np.take_along_axis(Pstack, tier[None, ...], axis=0)[0]       # (N,C): read each (b,k) at tier[b,k]
    mm = _macro_micro(Y, routed, n_classes)
    label_ratio = np.bincount(tier.ravel(), minlength=4) / tier.size      # label-level
    depth = tier.max(axis=1)                                              # deepest tier per sample = propagation depth
    depth_ratio = np.bincount(depth, minlength=4) / tier.shape[0]         # sample-level
    return mm, label_ratio, depth_ratio, routed                          # routed,(N,C) = for the per-client(LW) computation


def _collect_routed(clients, loaders, n_clients, n_classes, device):
    """For the given set of loaders, collect the per-exit sigmoid probabilities and the mean binary
    entropy used for client/edge/region routing. Returns: (P[list of 4 (N,C)], H1, H2, H3, Y, N, sizes). None if there is no data.
    sizes = number of samples per client (rows are stacked in client order and used to compute per-client metrics)."""
    P = [[] for _ in range(4)]          # per-exit sigmoid probabilities
    Hc = [[] for _ in range(3)]         # mean binary entropy for client/edge/region routing
    Y = []; sizes = []
    for c in range(n_clients):
        xb, lb, yb = loaders[c]
        if len(yb) == 0:
            continue
        sizes.append(int(len(yb)))
        full = clients[c]; full.eval_mode()
        with torch.no_grad():
            for s in range(0, len(yb), 512):
                xx = xb[s:s + 512].to(device); ll = lb[s:s + 512].to(device)
                c_out, c_aux = full.client(xx, ll)
                e_out, e_aux = full.edge(c_out, ll)
                r_out, r_aux = full.region(e_out, ll)
                cl_out = full.cloud(r_out, ll)
                for m, lg in enumerate((c_aux, e_aux, r_aux, cl_out)):
                    P[m].append(torch.sigmoid(lg).cpu().numpy())
                for m, lg in enumerate((c_aux, e_aux, r_aux)):
                    Hc[m].append(mean_binary_entropy(lg).cpu().numpy())
                Y.append(yb[s:s + 512].numpy())
    if not Y:
        return None
    P = [np.concatenate(p, 0) for p in P]
    H1, H2, H3 = (np.concatenate(h, 0) for h in Hc)
    Y = np.concatenate(Y, 0)
    return P, H1, H2, H3, Y, Y.shape[0], sizes


def _routed_at(P, H1, H2, H3, Y, N, e1, e2, e3, n_classes):
    """sample-wise routed metrics + tier ratios at the given (E_th)."""
    tier = _route_tiers(H1, H2, H3, e1, e2, e3)
    routed = np.stack(P, 0)[tier, np.arange(N)]           # (N,C) per-sample routed exit probabilities
    mm = _macro_micro(Y, routed, n_classes)
    ratio = np.bincount(tier, minlength=4) / N
    return mm, ratio


def _row_at(P, H1, H2, H3, Y, N, e1, e2, e3, n_classes, cloud0, tol):
    """routed metrics as a single-row dict for CSV/JSON."""
    mm, ratio = _routed_at(P, H1, H2, H3, Y, N, e1, e2, e3, n_classes)
    return {
        "e1": round(float(e1), 5), "e2": round(float(e2), 5), "e3": round(float(e3), 5),
        "routed_macro_auroc": round(mm[0], 5), "routed_micro_auroc": round(mm[1], 5),
        "routed_macro_auprc": round(mm[2], 5), "routed_micro_auprc": round(mm[3], 5),
        "client_ratio": round(float(ratio[0]), 4), "edge_ratio": round(float(ratio[1]), 4),
        "region_ratio": round(float(ratio[2]), 4), "cloud_ratio": round(float(ratio[3]), 4),
        "cloud_usage": round(float(ratio[3]), 4),
        "pass_tol": bool(mm[0] >= cloud0 - tol),
    }


def eth_routed_sweep(args, clients, sel_loaders, report_loaders, n_clients, n_classes, device):
    """phenotype routed eval + E_th sweep.
    Sweep and select candidates on sel_loaders, then report the final metrics once on report_loaders at the selected E_th.
      --eth_select_on test (default): the caller passes sel==report (both test_loaders) -> same as before.
      --eth_select_on val       : sel=val split from train, report=test -> avoids test-set threshold tuning
                                   (prevents tuning the operating point to test in label-shift/temporal experiments).
    Does not touch the existing evaluate_pheno (per-exit, threshold-free). Called only when --tune_eth."""
    same = report_loaders is sel_loaders
    sel_name = ("val" if args.val_frac > 0 else "test") if same else "val"
    rep_name = ("val" if args.val_frac > 0 else "test") if same else "test"

    sel = _collect_routed(clients, sel_loaders, n_clients, n_classes, device)
    if sel is None:
        print(f"[tune_eth] the selection ({sel_name}) set is empty, skipping the sweep"); return
    Ps, H1s, H2s, H3s, Ys, Ns, sizes_s = sel
    cloud_sel = _macro_micro(Ys, Ps[3], n_classes)        # sel cloud-only (for the tol threshold)
    print(f"\n=== [tune_eth] E_th selected on {sel_name} (N={Ns}) / final report on {rep_name} "
          f"(multi-label, final_local_adapt=0) ===")
    print(f"  [{sel_name}] cloud-only: macroAUROC={cloud_sel[0]:.4f} microAUROC={cloud_sel[1]:.4f} "
          f"macroAUPRC={cloud_sel[2]:.4f} microAUPRC={cloud_sel[3]:.4f}")

    # ---- generate candidates (1 if fixed E_th, else a percentile grid of sel entropy) ----
    if args.eth_fixed.strip():
        e1, e2, e3 = (float(x) for x in args.eth_fixed.split(","))
        cands = [(e1, e2, e3)]
        print(f"  [eth_fixed] fixed threshold ({e1:.4f},{e2:.4f},{e3:.4f})")
    else:
        pcts = np.linspace(args.eth_p_lo, args.eth_p_hi, args.eth_npct)
        e1c = np.percentile(H1s, pcts); e2c = np.percentile(H2s, pcts); e3c = np.percentile(H3s, pcts)
        cands = [(a, b, c) for a in e1c for b in e2c for c in e3c]
        print(f"  sweep on {sel_name}: {args.eth_npct} percentiles per exit"
              f"(p{args.eth_p_lo:g}~p{args.eth_p_hi:g}) -> {len(cands)} candidates")

    # ---- evaluate candidates on sel + select ----
    rows = [_row_at(Ps, H1s, H2s, H3s, Ys, Ns, e1, e2, e3, n_classes,
                    cloud_sel[0], args.eth_tolerance) for (e1, e2, e3) in cands]
    if args.eth_metric == "lw":
        Pstack_sel = np.stack(Ps, 0)
        for r, (e1, e2, e3) in zip(rows, cands):
            mm, _lab, depth, _ = _labelwise_routed(Pstack_sel, Ys, e1, e2, e3, n_classes)
            r["routed_lw_macro_auroc"] = round(float(mm[0]), 5)
            r["cloud_usage_depth"]     = round(float(depth[3]), 4)
            r["pass_tol"]              = bool(mm[0] >= cloud_sel[0] - args.eth_tolerance)
        passed = [r for r in rows if r["pass_tol"]]
        pool = passed if passed else rows
        best_sel = min(pool, key=lambda r: (r["cloud_usage_depth"], -r["routed_lw_macro_auroc"]))
        print(f"  [eth_metric=lw] selected by routed-LW: LW={best_sel['routed_lw_macro_auroc']:.4f} "
              f"depth-cloud={best_sel['cloud_usage_depth']:.4f}")
    else:
        passed = [r for r in rows if r["pass_tol"]]
        pool = passed if passed else rows
        best_sel = min(pool, key=lambda r: (r["cloud_usage"], -r["routed_macro_auroc"]))
    be1, be2, be3 = best_sel["e1"], best_sel["e2"], best_sel["e3"]

    # ---- lambda selection criterion: routed-LW pooled macroAUROC (on sel/val, at the selected E_th) ----
    # the driver picks the lambda combination by this value (val_selection.labelwise_macro_auroc).
    _, _, _, _sel_lwr = _labelwise_routed(np.stack(Ps, 0), Ys, be1, be2, be3, n_classes)
    sel_lw = _macro_micro(Ys, _sel_lwr, n_classes)        # (macro,micro,...)  sel routed-LW

    # ---- evaluate the selected E_th once on the report set (test) -> the number reported in the paper ----
    if same:
        Pr, H1r, H2r, H3r, Yr, Nr, sizes_r = Ps, H1s, H2s, H3s, Ys, Ns, sizes_s
        cloud_rep = cloud_sel
    else:
        rep = _collect_routed(clients, report_loaders, n_clients, n_classes, device)
        if rep is None:
            print(f"[tune_eth] the report ({rep_name}) set is empty, falling back to the sel metric"); 
            Pr, H1r, H2r, H3r, Yr, Nr, sizes_r = Ps, H1s, H2s, H3s, Ys, Ns, sizes_s; cloud_rep = cloud_sel
        else:
            Pr, H1r, H2r, H3r, Yr, Nr, sizes_r = rep
            cloud_rep = _macro_micro(Yr, Pr[3], n_classes)
            print(f"  [{rep_name}] cloud-only: macroAUROC={cloud_rep[0]:.4f} "
                  f"microAUROC={cloud_rep[1]:.4f} macroAUPRC={cloud_rep[2]:.4f} "
                  f"microAUPRC={cloud_rep[3]:.4f}")

    # the 'reporting' row re-evaluated on the report set at the selected E_th
    best = _row_at(Pr, H1r, H2r, H3r, Yr, Nr, be1, be2, be3, n_classes,
                   cloud_rep[0], args.eth_tolerance)

    # ---- label-wise routed: independent (sample,label) exits at the selected threshold (on the report set) ----
    Pstack = np.stack(Pr, 0)                                                # (4,Nr,C)
    lw_mm, lw_lr, lw_dr, lw_routed = _labelwise_routed(Pstack, Yr, be1, be2, be3, n_classes)
    # per-client macro-AUROC from the routed-LW probabilities (on the report set)
    pc_lw_au, pc_lw_n, pc_lw_lab = _per_client_macro_auroc(Yr, lw_routed, sizes_r)
    labelwise_info = {
        "labelwise_macro_auroc": round(lw_mm[0], 5),
        "labelwise_micro_auroc": round(lw_mm[1], 5),
        "labelwise_macro_auprc": round(lw_mm[2], 5),
        "labelwise_micro_auprc": round(lw_mm[3], 5),
        "labelwise_label_exit_ratio": [round(float(x), 4) for x in lw_lr],    # label-level
        "labelwise_sample_depth_ratio": [round(float(x), 4) for x in lw_dr],  # sample-level (propagation depth)
        "labelwise_cloud_usage": round(float(lw_dr[3]), 4),
        "labelwise_per_client_macro_auroc": round(float(pc_lw_au), 5),        # per-client (based on routed-LW)
        "labelwise_per_client_n": int(pc_lw_n),
    }

    tag = args.run_tag or ("mtsfl_" + args.task)
    csv_path = os.path.join(args.data_dir, f"{tag}_eth_search.csv")
    cols = ["e1", "e2", "e3", "routed_macro_auroc", "routed_micro_auroc",
            "routed_macro_auprc", "routed_micro_auprc", "client_ratio", "edge_ratio",
            "region_ratio", "cloud_ratio", "cloud_usage", "pass_tol"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()  # ignore extra keys
        for r in sorted(rows, key=lambda r: r["cloud_usage"]):                # the sweep is on sel
            w.writerow(r)
    best_json = os.path.join(args.data_dir, f"{tag}_best_eth.json")
    out = {"run_tag": tag,
           "eth_select_on": sel_name, "reported_on": rep_name,
           "eval_on": rep_name,                          # backward-compat: the report set
           "test_mode": args.test_mode, "labelshift_temp": args.labelshift_temp,
           "tolerance": args.eth_tolerance,
           "cloud_only_macro_auroc": round(cloud_rep[0], 5),   # report-set cloud-only
           "selected": best,                             # selected-E_th metrics on the report (test) set
           "labelwise": labelwise_info,
           "n_candidates": len(rows), "n_passed": len(passed)}
    if not same:                                         # when selecting on val, also record the val metrics for transparency
        out["val_selection"] = {
            "e1": be1, "e2": be2, "e3": be3,
            "routed_macro_auroc": best_sel["routed_macro_auroc"],        # sample-wise (val)
            "labelwise_macro_auroc": round(float(sel_lw[0]), 5),         # routed-LW pooled (val) = lambda selection criterion
            "cloud_usage": best_sel["cloud_usage"],
            "cloud_only_macro_auroc": round(cloud_sel[0], 5)}
    json.dump(out, open(best_json, "w"), indent=2, ensure_ascii=False)

    tail = "" if same else f"  [E_th selected on {sel_name}; the numbers below are reported on {rep_name}]"
    print(f"  [selected] e=({best['e1']:.4f},{best['e2']:.4f},{best['e3']:.4f})  "
          f"routed macroAUROC={best['routed_macro_auroc']:.4f} "
          f"(cloud-only {cloud_rep[0]:.4f}, Δ={best['routed_macro_auroc']-cloud_rep[0]:+.4f})  "
          f"macroAUPRC={best['routed_macro_auprc']:.4f}{tail}")
    print(f"  exit ratio ({rep_name}): client={best['client_ratio']:.1%} edge={best['edge_ratio']:.1%} "
          f"region={best['region_ratio']:.1%} cloud={best['cloud_usage']:.1%} "
          f"-> saved {1-best['cloud_usage']:.1%} of cloud usage")
    if not same:
        print(f"  (ref [{sel_name}] selection point: routed macroAUROC={best_sel['routed_macro_auroc']:.4f}, "
              f"cloud usage={best_sel['cloud_usage']:.1%})")
    # ---- label-wise results (compared on the same line as sample-wise / cloud-only) ----
    print(f"  [label-wise @ same E_th=({best['e1']:.4f},{best['e2']:.4f},{best['e3']:.4f})] "
          f"macroAUROC={lw_mm[0]:.4f} microAUROC={lw_mm[1]:.4f} macroAUPRC={lw_mm[2]:.4f}  "
          f"(sample-wise {best['routed_macro_auroc']:.4f}, cloud-only {cloud_rep[0]:.4f}, "
          f"Δ_vs_cloud={lw_mm[0]-cloud_rep[0]:+.4f})")
    print(f"    label-exit ratio   : client={lw_lr[0]:.1%} edge={lw_lr[1]:.1%} "
          f"region={lw_lr[2]:.1%} cloud={lw_lr[3]:.1%}   (= per-tier fraction of the 25-label predictions)")
    print(f"    sample-depth ratio : client={lw_dr[0]:.1%} edge={lw_dr[1]:.1%} "
          f"region={lw_dr[2]:.1%} cloud={lw_dr[3]:.1%}   "
          f"-> sample-level cloud usage {lw_dr[3]:.1%} (reduced by {1-lw_dr[3]:.1%})")
    print(f"  [per-client macro-AUROC (routed-LW, {rep_name})] {pc_lw_au:.4f}  "
          f"(over {pc_lw_n} clients)")
    if not same:
        print(f"  [lambda selection] {sel_name} routed-LW pooled macroAUROC = {sel_lw[0]:.4f}")
    print(f"  saved: {csv_path} , {best_json}")


if __name__ == "__main__":
    main()