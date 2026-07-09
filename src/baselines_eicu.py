"""
baselines_eicu.py -- training loops for the six comparison baselines on the eICU phenotype task.

Provides HierFL (hierarchical FedAvg of the whole model, edge@tau2 / region@tau3 / cloud@tau4),
Edge-based FL (edge-isolated FedAvg within each hospital, every tau2), HSFL (per-tier global
aggregation with per-round sync), Personalized FL (local + global with output mixing, global
FedAvg every tau4), LG-FedAvg (private body + globally aggregated head every tau4), and PHSFL
(features-only body FedAvg with periodic head fine-tuning). All share the data pipeline, optimizer,
cosine schedule, and evaluation with MT-SFL; selected via --baseline in train_eicu_mtsfl.py.
"""
import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import baselines_common as bc


# ============================================================================
# small helpers
# ============================================================================
def _opt(params, args, lr=None):
    params = list(params)
    lr = args.lr if lr is None else lr
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(params, lr=lr, weight_decay=args.weight_decay)


def _forward_cloud(model, x, lab):
    """4-tier forward -> (cloud_logits, (c_aux, e_aux, r_aux)). multilabel: raw logits."""
    c_out, c_aux = model.client(x, lab)
    e_out, e_aux = model.edge(c_out, lab)
    r_out, r_aux = model.region(e_out, lab)
    cl = model.cloud(r_out, lab)
    return cl, (c_aux, e_aux, r_aux)


def _train_epoch(model, loader, opt, device, mix_model=None):
    """One local pass, cloud-only BCE (multilabel). mix_model set -> PFL 0.5/0.5 logit mix."""
    model.train_mode()
    if mix_model is not None:
        mix_model.eval_mode()
    for x, lab, y in loader:
        if x.size(0) < 2:
            continue
        x, lab, y = x.to(device), lab.to(device), y.to(device)
        cl, _ = _forward_cloud(model, x, lab)
        if mix_model is not None:
            with torch.no_grad():
                gcl, _ = _forward_cloud(mix_model, x, lab)
            cl = 0.5 * cl + 0.5 * gcl
        loss = F.binary_cross_entropy_with_logits(cl, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def _train_all(ctx, opts, mix_model=None):
    """All clients do args.local_epochs inner passes (cloud-only)."""
    clients, train_loaders, device, args = ctx["clients"], ctx["train_loaders"], ctx["device"], ctx["args"]
    for c in range(ctx["n_clients"]):
        if c not in train_loaders:
            continue
        for _ in range(args.local_epochs):
            _train_epoch(clients[c], train_loaders[c], opts[c], device, mix_model=mix_model)


def _per_client_macro_auroc(Y, P, client_sizes):
    """per-client (macro-over-clients) macro-AUROC. Same definition as in evaluate_pheno.
    For each client, per-label AUROC on its own samples -> macro within the client -> average over clients.
    Labels that are single-class within a client are skipped; clients with no valid label are excluded.
    Returns: (mean_auroc, n_valid_clients, avg_valid_labels_per_client)."""
    offs = np.cumsum([0] + [int(s) for s in client_sizes])
    per_client, n_lab = [], []
    for i in range(len(client_sizes)):
        a, b = offs[i], offs[i + 1]
        yc, pc = Y[a:b], P[a:b]
        aus = []
        for k in range(yc.shape[1]):
            col = yc[:, k]
            if col.min() == col.max():
                continue
            aus.append(roc_auc_score(col, pc[:, k]))
        if aus:
            per_client.append(float(np.mean(aus))); n_lab.append(len(aus))
    if not per_client:
        return 0.0, 0, 0.0
    return float(np.mean(per_client)), len(per_client), float(np.mean(n_lab))


def _print_pheno_table(P, T, n_classes, detailed=True, client_sizes=None, src="cloud"):
    """Replicates evaluate_pheno's per-exit AUROC/AUPRC print (used for PFL mixed eval)."""
    if not T:
        print("  [eval] empty test set"); return {"cloud_macro_auroc": 0.0, "per_exit": {}}
    Tall = np.concatenate(T, 0)
    names = ["client", "edge", "region", "cloud"]
    macro = []
    for m in range(4):
        Pm = np.concatenate(P[m], 0); au = []; ap = []
        for k in range(n_classes):
            yk = Tall[:, k]
            if yk.min() == yk.max():
                continue
            au.append(roc_auc_score(yk, Pm[:, k])); ap.append(average_precision_score(yk, Pm[:, k]))
        yf, pf = Tall.ravel(), Pm.ravel(); ok = yf.min() != yf.max()
        macro.append({"macro_auroc": float(np.mean(au)) if au else 0.0,
                      "micro_auroc": float(roc_auc_score(yf, pf)) if ok else 0.0,
                      "macro_auprc": float(np.mean(ap)) if ap else 0.0,
                      "micro_auprc": float(average_precision_score(yf, pf)) if ok else 0.0})
    # baselines report the full model only -> per-exit / per-client prints are suppressed (see _log_eval headline)
    out = {"cloud_macro_auroc": macro[3]["macro_auroc"], "per_exit": {names[m]: macro[m] for m in range(4)}}
    if client_sizes is not None:
        P_pc = np.concatenate(P[3], 0)          # PFL: cloud slot == 0.5*local+0.5*global mix
        pc_au, pc_n, pc_lab = _per_client_macro_auroc(Tall, P_pc, client_sizes)
        out["per_client_macro_auroc"] = pc_au
        out["per_client_src"] = src
        out["per_client_n"] = pc_n
    return out


def _eval_pheno_mixed(ctx, gmodel, detailed=True):
    """PFL eval: cloud exit = 0.5*sigmoid(local) + 0.5*sigmoid(global) (same format)."""
    clients, test_loaders, device = ctx["clients"], ctx["test_loaders"], ctx["device"]
    P = [[] for _ in range(4)]; T = []; client_sizes = []
    for c in range(ctx["n_clients"]):
        xb, lb, yb = test_loaders[c]
        if len(yb) == 0:
            continue
        client_sizes.append(int(len(yb)))
        lo = clients[c]; lo.eval_mode(); gmodel.eval_mode()
        with torch.no_grad():
            for s in range(0, len(yb), 512):
                xx, ll = xb[s:s + 512].to(device), lb[s:s + 512].to(device)
                cl_l, (ca, ea, ra) = _forward_cloud(lo, xx, ll)
                cl_g, _ = _forward_cloud(gmodel, xx, ll)
                pmix = 0.5 * torch.sigmoid(cl_l) + 0.5 * torch.sigmoid(cl_g)
                for m, pr in enumerate((torch.sigmoid(ca), torch.sigmoid(ea), torch.sigmoid(ra), pmix)):
                    P[m].append(pr.cpu().numpy())
                T.append(yb[s:s + 512].numpy())
    return _print_pheno_table(P, T, ctx["n_classes"], detailed, client_sizes=client_sizes, src="mixed")


def _log_eval(ctx, epoch1, res, history):
    args = ctx["args"]
    head = res.get("cloud_macro_auroc", res.get("acc", 0.0))
    history.append({"epoch": epoch1, "cloud_macro_auroc": head, "per_exit": res.get("per_exit", {}),
                    "per_client_macro_auroc": res.get("per_client_macro_auroc"),
                    "per_client_src": res.get("per_client_src")})
    out = os.path.join(args.data_dir, f"{args.run_tag}_baseline.json")
    json.dump(history, open(out, "w"), indent=2)
    print(f"  >> [headline] epoch {epoch1}: full model macro-AUROC = {head:.4f}", flush=True)


# ============================================================================
# hierarchy convenience (members_by[2]=hospital->clients, [3]=region->clients, [4]=_global)
# ============================================================================
def _edges(ctx):    return ctx["members_by"][2]                  # {hospital: [clients]}
def _regions(ctx):  return ctx["members_by"][3]                  # {region:   [clients]}
def _global(ctx):   return ctx["members_by"][4]["_global"]       # [all clients]


# ============================================================================
# 1) HFL — hierarchical FedAvg of the whole model
# ============================================================================
def run_hfl(ctx):
    args, counts = ctx["args"], ctx["counts"]
    opts = [_opt(m.parameters(), args) for m in ctx["clients"]]
    # LR scheduler (optional): if --lr_sched cosine, apply CosineAnnealing to each client's optimizer.
    # baselines do not run the MT-SFL main loop, so we must step here for the LR to actually decay.
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    print(f"[baseline=HFL] hierarchical FedAvg (whole model) edge@{args.tau2}/region@{args.tau3}/cloud@{args.tau4}"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts)
        for scope in bc.hier_triggers_separate(ep, (args.tau2, args.tau3, args.tau4)):
            if scope == "edge":
                for _, mem in _edges(ctx).items():
                    for a in bc.ATTRS:
                        bc.fedavg_attr(ctx["clients"], mem, a, counts)
            elif scope == "region":
                for _, mem in _regions(ctx).items():
                    for a in bc.ATTRS:
                        bc.fedavg_attr(ctx["clients"], mem, a, counts)
            else:  # cloud / global
                for a in bc.ATTRS:
                    bc.fedavg_attr(ctx["clients"], _global(ctx), a, counts)
        for s in scheds:                       # cosine: decay LR every epoch (T_max=num_epochs)
            s.step()
        if args.eval_every and ep % args.eval_every == 0:
            print(f"[epoch {ep}] HFL eval:")
            _log_eval(ctx, ep, ctx["evaluate_pheno"](True), history)
    _final(ctx, history, ctx["evaluate_pheno"])


# ============================================================================
# 2) EBFL — edge-isolated FedAvg (within hospital) every round
# ============================================================================
def run_ebfl(ctx):
    args, counts = ctx["args"], ctx["counts"]
    opts = [_opt(m.parameters(), args) for m in ctx["clients"]]
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    print("[baseline=EBFL] edge-isolated FedAvg (whole model, within hospital, every round)"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    ebfl_period = args.tau2                                       # edge-isolated aggregation period = tau2(=3)
    print(f"[EBFL] edge-isolated FedAvg every {ebfl_period} epochs (tau2); eval right after aggregation", flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts)
        for s in scheds:
            s.step()
        if ep % ebfl_period == 0:
            for _, mem in _edges(ctx).items():
                for a in bc.ATTRS:
                    bc.fedavg_attr(ctx["clients"], mem, a, counts)
            print(f"[epoch {ep}] EBFL eval (right after agg):")
            _log_eval(ctx, ep, ctx["evaluate_pheno"](True), history)
    _final(ctx, history, ctx["evaluate_pheno"])


# ============================================================================
# 3) H-SFL — per-tier GLOBAL FedAvg at tiered frequency + per-round local sync
# ============================================================================
def run_hsfl(ctx):
    args, counts = ctx["args"], ctx["counts"]
    opts = [_opt(m.parameters(), args) for m in ctx["clients"]]
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    print(f"[baseline=H-SFL] per-tier global agg client@{args.tau2}/edge@{args.tau3}/region@{args.tau4} + per-round sync"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts)
        # per-round cost-free synchronization (Eq.3 in ref): edge within hospital,
        # region within region, cloud globally
        for _, mem in _edges(ctx).items():
            bc.fedavg_attr(ctx["clients"], mem, "edge", counts)
        for _, mem in _regions(ctx).items():
            bc.fedavg_attr(ctx["clients"], mem, "region", counts)
        bc.fedavg_attr(ctx["clients"], _global(ctx), "cloud", counts)
        # tiered global sub-model aggregation
        if ep % args.tau2 == 0:
            bc.fedavg_attr(ctx["clients"], _global(ctx), "client", counts)
        if ep % args.tau3 == 0:
            bc.fedavg_attr(ctx["clients"], _global(ctx), "edge", counts)
        if ep % args.tau4 == 0:
            bc.fedavg_attr(ctx["clients"], _global(ctx), "region", counts)
        for s in scheds:
            s.step()
        if args.eval_every and ep % args.eval_every == 0:
            print(f"[epoch {ep}] H-SFL eval:")
            _log_eval(ctx, ep, ctx["evaluate_pheno"](True), history)
    _final(ctx, history, ctx["evaluate_pheno"])


# ============================================================================
# 4) PH-SFL — features-only training (heads frozen), local body FedAvg, head fine-tune
# ============================================================================
def _phsfl_finetune_heads(ctx, k_epochs, lr=0.001):
    """Per-client cloud-head fine-tune. ALL bodies frozen (cuDNN-safe + correct PH-SFL):
    only cloud.head is in the backward graph, so eval-mode RNN bodies are never
    backpropped through."""
    clients, train_loaders, device, args, counts = (
        ctx["clients"], ctx["train_loaders"], ctx["device"], ctx["args"], ctx["counts"])
    for c in range(ctx["n_clients"]):
        if counts[c] == 0 or c not in train_loaders:
            continue
        bc.freeze_all_but_cloud_head(clients[c])          # freeze bodies, train only head
        opt = _opt(clients[c].cloud.head.parameters(), args, lr=lr)
        clients[c].eval_mode()                            # bodies eval (no dropout); head is Linear
        for _ in range(k_epochs):
            for x, lab, y in train_loaders[c]:
                if x.size(0) < 2:
                    continue
                x, lab, y = x.to(device), lab.to(device), y.to(device)
                cl, _ = _forward_cloud(clients[c], x, lab)   # frozen bodies -> only head gets grad
                loss = F.binary_cross_entropy_with_logits(cl, y)
                opt.zero_grad(); loss.backward(); opt.step()


def run_phsfl(ctx):
    args, counts = ctx["args"], ctx["counts"]
    clients = ctx["clients"]
    for m in clients:
        bc.freeze_all_heads(m)                                   # train bodies only
    opts = [_opt(bc.body_params(m), args) for m in clients]
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    k_temp = getattr(args, "phsfl_temp_tune", 1)
    k_final = getattr(args, "phsfl_final_tune", 10)
    print(f"[baseline=PH-SFL] features-only (heads frozen) local body FedAvg edge@{args.tau2}/region@{args.tau3}/cloud@{args.tau4}; "
          f"head fine-tune temp={k_temp} final={k_final}"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts)
        for scope in bc.hier_triggers_separate(ep, (args.tau2, args.tau3, args.tau4)):
            if scope == "edge":     # client+edge bodies within hospital
                for _, mem in _edges(ctx).items():
                    bc.fedavg_attr_exclude(clients, mem, "client", "head.", counts)
                    bc.fedavg_attr_exclude(clients, mem, "edge", "head.", counts)
            elif scope == "region":  # client+edge+region bodies within region
                for _, mem in _regions(ctx).items():
                    bc.fedavg_attr_exclude(clients, mem, "client", "head.", counts)
                    bc.fedavg_attr_exclude(clients, mem, "edge", "head.", counts)
                    bc.fedavg_attr_exclude(clients, mem, "region", "head.", counts)
            else:                    # all bodies globally
                for a in bc.ATTRS:
                    bc.fedavg_attr_exclude(clients, _global(ctx), a, "head.", counts)
        for s in scheds:
            s.step()
        if args.eval_every and ep % args.eval_every == 0:
            # temp head fine-tune -> eval -> restore frozen heads + body-train grads
            snaps = {c: copy.deepcopy(clients[c].cloud.head.state_dict()) for c in range(ctx["n_clients"])}
            _phsfl_finetune_heads(ctx, k_temp)            # freezes bodies, trains heads
            print(f"[epoch {ep}] PH-SFL eval (head temp-tune {k_temp}):")
            res = ctx["evaluate_pheno"](True)
            _log_eval(ctx, ep, res, history)
            for c in range(ctx["n_clients"]):
                clients[c].cloud.head.load_state_dict(snaps[c])
                bc.set_phsfl_body_train(clients[c])       # restore: bodies trainable, heads frozen
    # final personalization: full per-client head fine-tune (kept), then final eval
    print(f"\n=== PH-SFL final head fine-tune ({k_final} epochs) ===", flush=True)
    _phsfl_finetune_heads(ctx, k_final)
    _final(ctx, history, ctx["evaluate_pheno"])


# ============================================================================
# 5) PFL — flat global FedAvg + 0.5 local / 0.5 global output mixing
# ============================================================================
def run_pfl(ctx):
    args, counts = ctx["args"], ctx["counts"]
    clients = ctx["clients"]
    gmodel = bc.make_global_clone(clients[0])                    # shared global model
    opts = [_opt(m.parameters(), args) for m in clients]
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    print("[baseline=PFL] local+global, 0.5/0.5 output mix, global FedAvg (whole model) every round"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    pfl_period = args.tau4                                        # global FedAvg period = tau4(=12)
    print(f"[PFL] global FedAvg every {pfl_period} epochs (tau4); eval right after aggregation", flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts, mix_model=gmodel)                  # train locals against frozen global
        for s in scheds:
            s.step()
        if ep % pfl_period == 0:
            # global = |D|-weighted FedAvg of all locals (whole model) -> write into gmodel
            for a in bc.ATTRS:
                contrib = [c for c in _global(ctx) if counts[c] > 0]
                subs = [getattr(clients[c], a) for c in contrib]
                avg = bc.avg_state(subs, [counts[c] for c in contrib])
                bc.load_state(getattr(gmodel, a), avg)
            print(f"[epoch {ep}] PFL eval (right after agg, mixed output):")
            _log_eval(ctx, ep, _eval_pheno_mixed(ctx, gmodel, True), history)
    print("\n=== final eval ===", flush=True)
    _log_eval(ctx, args.num_epochs, _eval_pheno_mixed(ctx, gmodel, True), history)
    print("[baseline done]", flush=True)


# ============================================================================
# 6) LG-FedAvg — share only cloud .head globally; bodies private
# ============================================================================
def _fedavg_submodule_full(clients, sub, counts):
    """|D|-weighted FedAvg over the entire clients[i].<sub> submodule (block+head), broadcast in place.
    Only floating parameters/buffers are averaged; integer buffers are left unchanged."""
    import torch
    tot = float(sum(counts)) or 1.0
    w = [c / tot for c in counts]
    ref = getattr(clients[0], sub).state_dict()
    avg = {}
    for k, t0 in ref.items():
        if t0.is_floating_point():
            avg[k] = sum(getattr(clients[i], sub).state_dict()[k].float() * w[i]
                         for i in range(len(clients))).to(t0.dtype)
        else:
            avg[k] = t0.clone()
    for m in clients:
        getattr(m, sub).load_state_dict(avg, strict=True)


def run_lgfedavg(ctx):
    args, counts = ctx["args"], ctx["counts"]
    opts = [_opt(m.parameters(), args) for m in ctx["clients"]]
    # LR scheduler (optional): if --lr_sched cosine, apply CosineAnnealing to each client's optimizer and step every epoch.
    scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, args.num_epochs)) for o in opts]
              if getattr(args, "lr_sched", "none") == "cosine" else [])
    history = []
    print("[baseline=LG-FedAvg] private body + global cloud.head FedAvg every round"
          + (f"  | LR schedule = cosine over {args.num_epochs} ep" if scheds else ""), flush=True)
    agg_period = args.tau4                                        # head aggregation period = same as the cloud tier (tau4)
    print(f"[LG-FedAvg] cloud.head FedAvg every {agg_period} epochs (tau4); eval right after aggregation", flush=True)
    for ep in range(1, args.num_epochs + 1):
        _train_all(ctx, opts)                                    # full local training
        for s in scheds:                                         # cosine: decay LR every epoch
            s.step()
        if ep % agg_period == 0:                                 # global head FedAvg only every tau4=12
            if os.environ.get("LGFEDAVG_AGG", "head") == "cloud":
                _fedavg_submodule_full(ctx["clients"], "cloud", counts)   # entire cloud block + head
            else:
                bc.fedavg_attr_module(ctx["clients"], _global(ctx), "cloud", "head", counts)
            print(f"[epoch {ep}] LG-FedAvg eval (right after head agg):")
            _log_eval(ctx, ep, ctx["evaluate_pheno"](True), history)
    _final(ctx, history, ctx["evaluate_pheno"])


# ============================================================================
def _final(ctx, history, eval_fn):
    print("\n=== final eval ===", flush=True)
    _log_eval(ctx, ctx["args"].num_epochs, eval_fn(True), history)
    print("[baseline done]", flush=True)


_DISPATCH = {"hfl": run_hfl, "ebfl": run_ebfl, "hsfl": run_hsfl,
             "phsfl": run_phsfl, "pfl": run_pfl, "lgfedavg": run_lgfedavg}


def run_baseline(name, ctx):
    if not ctx["is_pheno"]:
        raise SystemExit("[baseline] eICU baselines are phenotype-only (diagnosis excluded).")
    name = name.lower()
    if name not in _DISPATCH:
        raise SystemExit(f"[baseline] unknown '{name}', choose from {list(_DISPATCH)}")
    print(f"\n########## eICU phenotype baseline: {name.upper()} "
          f"(rnn={ctx['args'].rnn}, run_tag={ctx['args'].run_tag}) ##########", flush=True)
    _DISPATCH[name](ctx)