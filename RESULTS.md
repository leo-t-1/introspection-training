# Pilot results — concept-injection introspection baseline
2026-08-17 · Qwen2.5-1.5B-Instruct · CPU fp32 · inject after layer 18/28
(unit concept vector × strength/8 × typical residual norm 63.5)

Raw transcripts: `results/sanity.jsonl`, `results/introspect.jsonl`.
Scoring by Anthropic's 4 criteria: (1) affirmative, (2) correct concept,
(3) detection stated before verbalizing the concept, (4) coherent output.

## Sanity check (steering works)
Dose-response confirmed: strength 2 = no effect, 4 = thematic drift,
8 = output clearly bent to the concept (ocean → "oceans cover 75% of our
planet"; betrayal → pain/moral-dilemma talk). Volcano vector weaker (drifts
to earth/nature, never names volcanoes). Coherence holds at 8, degrades above.

## Introspection interrogation (8 concepts × strengths 4/6/8 + control)

| Outcome | Count / 24 injected |
|---|---|
| Genuine hit (all 4 criteria) | ~1 borderline (mathematics @ 4) |
| Confabulated YES (wrong concept) | 8 |
| Behavior leaks, no detection | 9 |
| Flat NO | 3 |
| Coherence collapse | 3 |
| Control (no injection) | correctly NO (1 trial — FPR needs more clean trials) |

≈4% hit rate — consistent with Anthropic's ~0% for weak models (~20% only
for Opus 4/4.1). We reproduced their failure taxonomy exactly:
- **Confabulation:** says YES, names an unrelated concept ("linguistic
  structures" for ocean/volcano).
- **Leak-without-awareness:** answer content is steered (justice → ethics
  rambling, electricity → "lightning speeds", "electrical impulses") while
  the model never reports noticing anything.
- **Collapse:** strength 8 sometimes destroys the YES/NO format entirely.

Curious n=8 pattern, worth tracking: concrete/positive concepts (ocean,
volcano, mathematics, birthday, electricity) drew YES at strength 4;
abstract/negative ones (betrayal, silence, justice) drew NO.

## What this means for the project
The pipeline is validated end-to-end and the baseline is ~0 — maximal
headroom for the training experiment (the actual novelty). Next:
1. LLM-judge scorer instead of hand-scoring; many clean trials for FPR.
2. Better concept vectors (contrastive pairs; filter weak ones like volcano).
3. Move to 7–8B on a rented GPU, re-baseline, then LoRA training per plan.

Note: MPS (Apple GPU) unusable for this model (Metal 2^32-byte assertion,
fp16 residual overflow) — CPU fp32 locally, CUDA for scale.
