"""
baselines_common.py -- shared utilities for the baseline training loops.

Optimizer/scheduler construction, |D_k|-weighted FedAvg aggregation helpers, and the evaluation
routines used by baselines_eicu.py.
"""
import copy
import torch
import torch.nn as nn

ATTRS = ["client", "edge", "region", "cloud"]


# --------------------------------------------------------------------------------------
# FedAvg over a list of nn.Modules (returns averaged state_dict on CPU)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def avg_state(modules, weights=None):
    """|D|-weighted mean of state_dicts, accumulated on CPU float32 (OOM-safe).
    weights None -> uniform mean. Returns None if modules is empty."""
    if not modules:
        return None
    if weights is None:
        weights = [1.0] * len(modules)
    W = float(sum(weights)) or 1.0
    acc = None
    for m, w in zip(modules, weights):
        f = w / W
        sd = m.state_dict()
        if acc is None:
            acc = {k: v.detach().to("cpu", torch.float32) * f for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k].add_(v.detach().to("cpu", torch.float32), alpha=f)
    return acc


@torch.no_grad()
def load_state(module, sd):
    """Load a (CPU) state_dict into module, matching its device/dtype per tensor."""
    tgt = module.state_dict()
    module.load_state_dict({k: sd[k].to(tgt[k].device, dtype=tgt[k].dtype) for k in tgt})


@torch.no_grad()
def fedavg_attr(clients, member_ids, attr, counts=None):
    """FedAvg getattr(clients[c], attr) over contributors (counts>0), broadcast to ALL
    member_ids (cold clients receive but do not contribute — matches MT-SFL).
    clients: list|dict.  attr in ATTRS.  counts: dict{id:int} or None (uniform)."""
    if counts is not None:
        contrib = [c for c in member_ids if counts.get(c, 0) > 0]
    else:
        contrib = list(member_ids)
    if not contrib:
        return
    subs = [getattr(clients[c], attr) for c in contrib]
    w = [counts[c] for c in contrib] if counts is not None else None
    avg = avg_state(subs, w)
    for c in member_ids:
        load_state(getattr(clients[c], attr), avg)


@torch.no_grad()
def fedavg_attr_exclude(clients, member_ids, attr, exclude_prefix="head.", counts=None):
    """FedAvg a tier but write back only keys NOT starting with exclude_prefix
    (PH-SFL: average the feature body, keep each client's frozen head)."""
    if counts is not None:
        contrib = [c for c in member_ids if counts.get(c, 0) > 0]
    else:
        contrib = list(member_ids)
    if not contrib:
        return
    subs = [getattr(clients[c], attr) for c in contrib]
    w = [counts[c] for c in contrib] if counts is not None else None
    avg = avg_state(subs, w)
    keep = {k: v for k, v in avg.items() if not k.startswith(exclude_prefix)}
    for c in member_ids:
        m = getattr(clients[c], attr)
        tgt = m.state_dict()
        new = {k: (keep[k].to(tgt[k].device, dtype=tgt[k].dtype) if k in keep else tgt[k]) for k in tgt}
        m.load_state_dict(new)


@torch.no_grad()
def fedavg_attr_module(clients, member_ids, attr, submod_name, counts=None):
    """Like fedavg_attr but only the named sub-submodule (e.g. attr='cloud',
    submod_name='head' for LG-FedAvg head sharing; or 'features'-style bodies)."""
    if counts is not None:
        contrib = [c for c in member_ids if counts.get(c, 0) > 0]
    else:
        contrib = list(member_ids)
    if not contrib:
        return
    subs = [getattr(getattr(clients[c], attr), submod_name) for c in contrib]
    w = [counts[c] for c in contrib] if counts is not None else None
    avg = avg_state(subs, w)
    for c in member_ids:
        load_state(getattr(getattr(clients[c], attr), submod_name), avg)


# --------------------------------------------------------------------------------------
# Freeze / param helpers  (PH-SFL: train bodies only, heads frozen; LG: head isolation)
# --------------------------------------------------------------------------------------
def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all_heads(model):
    """Freeze the final .head of every tier (PH-SFL main phase)."""
    for a in ATTRS:
        set_requires_grad(getattr(model, a).head, False)


def freeze_all_but_cloud_head(model):
    """PH-SFL head fine-tune: freeze EVERY param, then unfreeze only cloud.head.
    Critical on GPU: if bodies aren't frozen, autograd backprops through the cuDNN
    RNN bodies which raises 'cudnn RNN backward can only be called in training mode'
    when the model is in eval mode. Freezing bodies cuts their grad so only the head
    (a Linear) is in the backward graph — safe in eval mode, and the correct PH-SFL
    semantics (personalize the head over frozen features)."""
    for a in ATTRS:
        set_requires_grad(getattr(model, a), False)
    set_requires_grad(model.cloud.head, True)


def set_phsfl_body_train(model):
    """Restore PH-SFL main-phase grad state: all bodies trainable, all heads frozen."""
    for a in ATTRS:
        set_requires_grad(getattr(model, a), True)
    freeze_all_heads(model)


def body_params(model):
    """All trainable (non-head) params across the 4 tiers — for PH-SFL optimizer."""
    head_ids = set()
    for a in ATTRS:
        for p in getattr(model, a).head.parameters():
            head_ids.add(id(p))
    return [p for p in model.parameters() if id(p) not in head_ids]


def head_module(model):
    """The final classifier head used for prediction (cloud tier)."""
    return model.cloud.head


# --------------------------------------------------------------------------------------
# Aggregation schedule helpers
# --------------------------------------------------------------------------------------
def hier_triggers_separate(epoch1, tau):
    """HFL-style SEPARATE-if schedule (lower scopes also fire on higher-scope epochs).
    epoch1 = 1-based. tau=(t_edge, t_region, t_cloud). Returns ordered list of scopes
    due this epoch, edge->region->cloud (progressive widening, widest wins)."""
    t2, t3, t4 = tau
    due = []
    if epoch1 % t2 == 0:
        due.append("edge")
    if epoch1 % t3 == 0:
        due.append("region")
    if epoch1 % t4 == 0:
        due.append("cloud")
    return due


def make_global_clone(model):
    """Deep copy a tier model (PFL global model). Same architecture, on same device."""
    return copy.deepcopy(model)