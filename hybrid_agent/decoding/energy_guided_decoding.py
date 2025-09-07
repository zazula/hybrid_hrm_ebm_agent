
import torch
from torch import nn

@torch.no_grad()
def ebm_rerank(candidates_idx, ebm_model, feature_vec=None, length_penalty=0.2):
    """Rerank complete sequence candidates via EBM energy + length penalty.
    candidates_idx: (B, K, L)
    feature_vec:    (B, K, F) optional rule-based penalties
    Returns: best indices (B,), scores (B,K)
    """
    B, K, L = candidates_idx.shape
    flat = candidates_idx.view(B*K, L)
    feat = feature_vec.view(B*K, -1) if feature_vec is not None else None
    energies = ebm_model.energy(flat, feat)  # (B*K,)
    lengths = (flat != 0).sum(dim=-1).float()
    scores = -energies - length_penalty * lengths
    scores = scores.view(B, K)
    best = scores.argmax(dim=-1)
    return best, scores

@torch.no_grad()
def logit_shaping_with_energy(step_logits, partial_idx, ebm_model, alpha=1.0):
    """Simple logit shaping: subtract energy gradient estimate w.r.t. last token.
    This is a toy, finite-difference approximation to keep the dependency light.
    """
    B, V = step_logits.shape
    device = step_logits.device
    eps = 1e-4
    # probe a few top tokens for local slope
    topk = torch.topk(step_logits, k=min(32, V), dim=-1).indices  # (B, k)
    shaped = step_logits.clone()
    for b in range(B):
        base_seq = partial_idx[b:b+1]
        base_e = ebm_model.energy(base_seq).item()
        for t in topk[b]:
            seq = torch.cat([base_seq, t.view(1,1)], dim=-1)
            e = ebm_model.energy(seq).item()
            grad = (e - base_e) / eps
            shaped[b, t] -= alpha * grad
    return shaped
