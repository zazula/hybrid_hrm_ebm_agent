
from dataclasses import dataclass

@dataclass
class HRMConfig:
    d_model:int=512
    d_plan:int=512
    n_layers:int=4
    n_heads:int=8
    dropout:float=0.1
    plan_update_interval:int=8  # steps between high-level updates

@dataclass
class LLMConfig:
    vocab_size:int=32000
    d_model:int=1024
    n_layers:int=16
    n_heads:int=16
    max_seq_len:int=4096
    dropout:float=0.1

@dataclass
class EBMConfig:
    d_model:int=768
    n_layers:int=8
    n_heads:int=12
    feature_lambda:float=0.5  # weight on rule-based features
    temperature:float=1.0

@dataclass
class DecodingConfig:
    beam_size:int=4
    max_steps:int=128
    ebm_lambda:float=1.0    # energy weight
    length_penalty:float=0.2
    min_p:float=1e-6

@dataclass
class HybridConfig:
    llm: LLMConfig = LLMConfig()
    hrm: HRMConfig = HRMConfig()
    ebm: EBMConfig = EBMConfig()
    decode: DecodingConfig = DecodingConfig()
