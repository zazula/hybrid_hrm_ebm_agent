import torch
import types

from hybrid_agent.decoding.energy_guided_decoding import ebm_rerank, logit_shaping_with_energy


class DummyEBM:
    def __init__(self, token_weight: float = 1.0, feat_weight: float = 1.0):
        self.token_weight = token_weight
        self.feat_weight = feat_weight

    @torch.no_grad()
    def energy(self, seq_idx: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        # energy = token_weight * sum(token_ids) + feat_weight * sum(features)
        # Accept shapes: (B, L) or (1, L)
        if seq_idx.dim() == 1:
            seq_idx = seq_idx.unsqueeze(0)
        tok_term = self.token_weight * seq_idx.float().sum(dim=-1)
        if feat is not None:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            feat_term = self.feat_weight * feat.float().sum(dim=-1)
        else:
            feat_term = torch.zeros_like(tok_term)
        return tok_term + feat_term


def test_ebm_rerank_prefers_lower_energy_after_length_penalty():
    torch.manual_seed(0)
    B, K, L = 2, 3, 4
    # Construct candidates so that candidate 1 has higher token ids (higher energy)
    cands = torch.tensor(
        [
            # batch 0
            [
                [1, 1, 0, 0],  # sum=2
                [3, 0, 0, 0],  # sum=3 (worse)
                [1, 0, 0, 0],  # sum=1 (best)
            ],
            # batch 1
            [
                [0, 0, 0, 0],  # sum=0 (best)
                [2, 2, 0, 0],  # sum=4 (worse)
                [1, 1, 1, 0],  # sum=3
            ],
        ],
        dtype=torch.long,
    )
    feat = torch.zeros(B, K, 2)
    ebm = DummyEBM(token_weight=1.0, feat_weight=0.0)

    best, scores = ebm_rerank(cands, ebm, feature_vec=feat, length_penalty=0.1)

    # Expect indices of lowest energy (accounting for small length penalty)
    # batch 0 best is candidate 2 (index 2, sum=1)
    # batch 1 best is candidate 0 (index 0, sum=0)
    assert best.tolist() == [2, 0]
    assert scores.shape == (B, K)


def test_logit_shaping_penalizes_high_energy_tokens():
    # Create a simple partial seq [1, 1]; appending token 9 should incur high energy
    step_logits = torch.zeros(1, 20)  # all zeros (equal logits)
    partial = torch.tensor([[1, 1]], dtype=torch.long)

    class PenalizeNine(DummyEBM):
        @torch.no_grad()
        def energy(self, seq_idx: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
            if seq_idx.dim() == 1:
                seq_idx = seq_idx.unsqueeze(0)
            # Base: sum(token_ids)
            base = seq_idx.float().sum(dim=-1)
            # If last token is 9, add a large bump
            bump = (seq_idx[:, -1] == 9).float() * 5.0
            return base + bump

    ebm = PenalizeNine()
    shaped = logit_shaping_with_energy(step_logits.clone(), partial, ebm, alpha=1.0)

    # Token 9 should be decreased relative to baseline due to high energy gradient
    assert shaped[0, 9] < step_logits[0, 9]
    # Non-penalized token like 1 should not be penalized as much
    assert shaped[0, 1] >= shaped[0, 9]
