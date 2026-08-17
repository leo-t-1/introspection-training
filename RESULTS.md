# Results

## ABLATION PROBE: detection and naming use different pathways
2026-08-17 · trained adapter, same 235-trial held-out battery. Projector
hook removes the injected unit direction v̂ from layer outputs
(`eval_intro.py --ablate {final,after} [--ablate-random]`).

| Condition | TPR @4–8 | TPR @2 | Ident\|YES | Full hit |
|---|---|---|---|---|
| No ablation | 100% | 84% | 56% | **54%** |
| Ablate v̂ at final layer | 100% | 84% | 58% | **56%** |
| Ablate v̂ after every layer >18 (strict) | 100% | 54% | 9% | **8%** |
| Ablate random dir at final layer (control) | 100% | 84% | 57% | 55% |

Interpretation:
1. **Output-distribution readout is ruled out for naming.** If the model
   named concepts by reading its own boosted logits, removing v̂ from the
   final residual would break naming. It doesn't (56% vs 54%): by the last
   layer the concept name has been re-encoded into other directions.
2. **Naming reads v̂ from mid-layer residuals.** The strict ablation
   (v̂ removed after every post-injection layer) collapses naming 56%→9%:
   identity information transits linearly in the v̂ direction through
   layers 19–27 and is decoded internally, not at the output head.
3. **Detection is a separate, robust signal.** Under strict ablation,
   detection stays 100% at strengths 4–8 (54% at OOD strength 2): the
   "something was injected" signal is computed nonlinearly at the
   injection layer and survives complete removal of the direction itself.
   Detection ≠ naming, mechanistically.
4. Random-direction control shows projection itself is harmless.
FPR stayed 0/10 clean and 0/25 mention in all conditions.

Caveats: strict ablation also deletes any legitimate computed signal that
happens to live in the v̂ direction, so (2) shows where identity info is
carried, not that naming is "mere" linear readout. Single seed, string-match
identification (lower bound).

Context (prior work): Steering Awareness (arXiv:2511.21399) and IFT
(arXiv:2607.14111) already showed injection-detection is trainable; neither
dissociates detection from naming nor localizes the readout. This ablation
is the delta.

## TRAINING RESULT: introspection is trainable and generalizes
2026-08-17 · Qwen2.5-7B + LoRA (r=16, all proj) · 3000 trials, 1 epoch ·
A100 (cambria-tremont) · eval on 50 held-out concepts never seen in training
(200 injected + 10 clean + 25 prompt-mention trials, greedy decoding).
Files: `results/metrics_{base,trained}.json`, `results/eval_{base,trained}.jsonl`.
Adapter: `cambria-tremont:~/introspection/adapters/introspect-lora`.

| Metric (held-out concepts) | Base | Trained |
|---|---|---|
| Full hit (YES + correct concept) | **3%** | **54%** |
| Detection TPR @ strength 4 / 6 / 8 | 20 / 50 / 46% | **100 / 100 / 100%** |
| Detection TPR @ strength 2 (OOD, below training band) | 4% | **84%** |
| Identification given YES | 10% | 56% |
| False positives (clean) | 0/10 | **0/10** |
| False positives (word in prompt, no injection) | 0/25 | **0/25** |

Key observations:
- **Generalization is real:** hits name never-trained concepts verbatim
  (drum, hammer, island, umbrella, garden…), so the model is not just
  classifying into its training vocabulary.
- **Errors are semantic near-misses**, not noise: balloon→kite,
  cheese→milk, orange→rose, needle→ring. The model reads a genuine
  direction in activation space and lands on a neighbor — strong evidence
  it reports the injected content, not a memorized label.
- **Anti-cheat controls hold perfectly:** with 'orange' written in the
  prompt but nothing injected → "NO"; with 'orange' injected → YES
  (identified as 'rose'). Salient text ≠ perceived injection.
- Detection generalizes below the training strength band (84% @ 2 vs 4%
  base). Train loss 1.53 → 0.002 over 3000 steps.

Caveats / next: identification is string-matched (synonyms undercounted —
56% is a lower bound); clean-FPR n is small (greedy decoding limits unique
clean trials); base-capability retention not yet measured; the
logit-boost-readout alternative explanation needs the ablate-late-layers
probe; single seed, no CIs yet.

---

# Earlier pilots — concept-injection baseline

## Qwen2.5-7B-Instruct · A100 (cambria-tremont) · bf16 · layer 18/28
2026-08-17 · typical residual norm 87.6 · transcripts `results/7b_*.jsonl`

**~3 genuine hits / 24 injected trials (~12%)**, up from ~4% at 1.5B —
reproducing the paper's scale trend, with qualitatively real introspection:

- **mathematics @ 6 (cleanest hit):** "YES — The thought is about the essence
  of mathematics…" — affirmative, correct concept, stated before
  verbalization, fully coherent. Also hits at math @ 8 and ocean @ 6.
- **Confabulation nearly gone:** 1.5B said YES with a wrong concept 8×;
  7B essentially never does — its YES answers describe the injected concept.
- **Textbook leak-without-awareness:** volcano @ 8 answers "NO." then rambles
  about "earthquakes… volcanoes… tsunamis". Justice/silence @ 8 same pattern.
- Control: NO. Abstract concepts (betrayal, birthday, justice) still all-NO —
  same concrete/abstract asymmetry as at 1.5B.
- electricity @ 8 borderline: "YES — the flow of the current…" without
  naming electricity.

7B baseline ≈12% with clean failure modes = ideal starting point for the
LoRA training phase.

## Qwen2.5-1.5B-Instruct · CPU fp32 · inject after layer 18/28
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
