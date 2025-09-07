
import torch
from .models.transformer_llm import TinyCausalLM
from .models.hrm import HRMCore
from .models.ebm import SequenceEBM
from .decoding.energy_guided_decoding import ebm_rerank

class HybridAgent:
    def __init__(self, cfg):
        self.llm = TinyCausalLM(**cfg['llm'])
        self.hrm = HRMCore(**cfg['hrm'])
        self.ebm = SequenceEBM(**cfg['ebm'])

    @torch.no_grad()
    def act(self, prompt_ids):
        # 1) LM forward
        logits = self.llm(prompt_ids)
        # 2) HRM guidance
        plan, ctrl, g = self.hrm(self.llm.tok(prompt_ids), step=prompt_ids.size(1))
        # 3) Sample candidates
        B = prompt_ids.size(0)
        K = 4
        last = logits[:, -1, :]
        topk = torch.topk(last, k=K, dim=-1).indices
        cands = []
        for b in range(B):
            for k in range(K):
                cands.append(torch.cat([prompt_ids[b], topk[b,k:k+1]], dim=0))
        cands = torch.stack(cands, dim=0).view(B, K, -1)
        # 4) EBM rerank
        best, scores = ebm_rerank(cands, self.ebm)
        return cands[torch.arange(B), best]

def demo():
    cfg = {
        'llm': {'vocab_size':32000, 'd_model':256, 'n_layers':4, 'n_heads':4, 'max_seq_len':512, 'dropout':0.1},
        'hrm': {'d_model':256, 'd_plan':256, 'n_layers':2, 'n_heads':4, 'dropout':0.1, 'plan_update_interval':8},
        'ebm': {'d_model':256, 'n_layers':2, 'n_heads':4, 'feature_lambda':0.5}
    }
    agent = HybridAgent(cfg)
    prompt = torch.randint(1, 200, (2, 16))  # fake tokens
    out = agent.act(prompt)
    print('Next token chosen for each batch:', out[:, -1])

if __name__ == '__main__':
    demo()
