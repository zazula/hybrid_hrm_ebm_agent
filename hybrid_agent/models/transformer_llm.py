
import torch, torch.nn as nn, math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.register_buffer('mask', None, persistent=False)

    def forward(self, x):
        L = x.size(1)
        if self.mask is None or self.mask.size(0) < L:
            # causal mask (L, L)
            mask = torch.full((L, L), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            self.mask = mask
        out, _ = self.attn(x, x, x, attn_mask=self.mask[:L, :L])
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, n_layers=16, n_heads=16, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def step(self, idx, temperature=1.0):
        logits = self.forward(idx)[:, -1, :]
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        return next_tok
