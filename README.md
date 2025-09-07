
# Hybrid HRM+LLM+EBM Agent (skeleton)

A reference skeleton that blends a code-centric Transformer with a Hierarchical Reasoning Module (HRM) and a Sequence Energy-Based Model (EBM) for energy-guided planning/decoding. 
Designed for autonomous computer use and coding tasks.

## Layout
- `hybrid_agent/`
  - `models/transformer_llm.py` — token LM (decoder) wrapper
  - `models/hrm.py` — hierarchical high/low-timescale planner
  - `models/ebm.py` — bi-directional sequence EBM scoring
  - `decoding/energy_guided_decoding.py` — EBM rerank & logit shaping
  - `tools/{code_tools.py,os_tools.py}` — execution tools & validators
  - `envs/{osworld_wrapper.py,swebench_wrapper.py,webarena_wrapper.py}` — env adapters
  - `train/{train_sft.py,train_rl.py,train_ebm.py}` — training entry points (stubs)
  - `eval/{eval_osworld.py,eval_swebench.py,eval_gaia.py}` — evaluation stubs
  - `run_agent.py` — demo loop (CPU-safe)

This is a compile-ready scaffold; fill the TODOs with your datasets and models.
