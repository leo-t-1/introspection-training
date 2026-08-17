# Introspection Training

Can you *train* an LLM to be better at introspection? This project reproduces
Anthropic's concept-injection experiment ("Emergent Introspective Awareness
in LLMs", Lindsey 2026, arXiv:2601.01828) on small open models, then
fine-tunes directly on the task — which the original paper did not attempt.

## Status

- [x] Pipeline reproduced locally (Qwen2.5-1.5B, CPU): steering works,
      baseline introspection ≈4% with Anthropic's exact failure modes.
      See [RESULTS.md](RESULTS.md).
- [ ] LLM-judge scorer + proper false-positive measurement
- [ ] 7–8B baseline on rented GPU
- [ ] LoRA training on injection trials (the novel part) — see [PLAN.md](PLAN.md)

## Run it

```bash
python3 -m venv --system-site-packages .venv   # reuses system torch
.venv/bin/pip install transformers accelerate
.venv/bin/python inject.py --mode sanity      # does steering work?
.venv/bin/python inject.py --mode introspect  # the experiment
```

Trials land in `results/*.jsonl`. On Apple Silicon use the default
`--device cpu` (MPS crashes on this model family; fp16 overflows).

## Method in one paragraph

A concept vector is the activation of "Tell me about {word}" minus the mean
over 30 baseline words, taken at ~2/3 model depth. It is added to the
residual stream at every position via a forward hook while the model is
asked whether it notices an injected thought and what it is. Ground truth is
causal — we know what was injected — so introspective accuracy is directly
scoreable (and trainable).
