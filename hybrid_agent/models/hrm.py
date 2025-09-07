
import torch, torch.nn as nn

class HRMCore(nn.Module):
    """
    Two-timescale recurrent core:
      - slow planner state g_t (updates every K steps)
      - fast worker state  h_t (updates each step conditioned on g_t)
    Outputs a 'plan token' and a control vector used to steer decoding/tools.
    """
    def __init__(self, d_model=512, d_plan=512, n_layers=4, n_heads=8, dropout=0.1, plan_update_interval=8):
        super().__init__()
        self.plan_update_interval = plan_update_interval
        self.planner = nn.GRU(input_size=d_model, hidden_size=d_plan, num_layers=1, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model+d_plan, nhead=n_heads, dropout=dropout, batch_first=True)
        self.worker = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.plan_head = nn.Linear(d_plan, d_plan)
        self.ctrl_head = nn.Linear(d_model+d_plan, d_plan)

    def forward(self, x, g=None, step:int=0):
        # x: (B, L, d_model) token features from LM
        B, L, D = x.shape
        update = (step % self.plan_update_interval) == 0
        if g is None:
            g = torch.zeros(1, B, self.plan_head.in_features, device=x.device)

        if update:
            # aggregate last K steps (or all available) via mean-pool
            pooled = x.mean(dim=1, keepdim=True)  # (B,1,D)
            _, g_next = self.planner(pooled, g)   # (1,B,d_plan)
        else:
            g_next = g

        g_expand = g_next.permute(1,0,2).expand(B, L, -1)  # (B,L,d_plan)
        hw_in = torch.cat([x, g_expand], dim=-1)
        hw = self.worker(hw_in)
        ctrl = self.ctrl_head(hw)[:, -1, :]  # (B, d_plan)
        plan = self.plan_head(g_next.squeeze(0))  # (B, d_plan)
        return plan, ctrl, g_next
