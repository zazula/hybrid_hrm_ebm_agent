
import torch, torch.nn as nn

class SequenceEBM(nn.Module):
    """Bi-directional transformer scoring sequence energy.
    Energy is a scalar per sequence; lower is better.
    """
    def __init__(self, d_model=768, n_layers=8, n_heads=12, feature_lambda=0.5):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.embed = nn.Embedding(32000, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.feature_lambda = feature_lambda

    def forward(self, idx, feature_vec=None):
        x = self.embed(idx)                    # (B,L,D)
        h = self.encoder(x)                    # (B,L,D)
        pooled = h.mean(dim=1)                 # (B,D)
        e_model = self.head(pooled).squeeze(-1)  # (B,)
        if feature_vec is not None:
            e_feat = (feature_vec).sum(dim=-1)  # (B,)
            return e_model + self.feature_lambda * e_feat
        return e_model

    def energy(self, idx, feature_vec=None):
        return self.forward(idx, feature_vec)
