# Adversarial referee report (internal red-team, 2026-08-17)

Verdict: reject in current form. Response matrix first; full report below.

## Response matrix

| Issue | Status |
|---|---|
| F1 naming = live steering readout, not report | **Controls running**: base-model prefill naming; prompt-only injection (injection cut at generation start) |
| F2 dissociation guaranteed by geometry; detection = magnitude alarm? | **Controls running**: random-vector injections (yes-rate), strict-ablation random-direction control, low-strength (0.5/1) threshold mapping |
| F3 FPR measured on memorized training prompts | **Fixed**: v2 battery with 6 never-trained templates, 54 adversarial clean prompts, naturalistic mentions — running |
| F4 base 3% confounded by format | **Fixed**: logprob AUROC (YES−NO margin) for base — running |
| M1 train-vocab bias in names | **Measured**: 56% correct heldout; 83% of *errors* are TRAIN words; but 14/192 emissions outside both vocabs and semantically apt (island→beach, seashell→pearl, eclipse→editor/debugger — the vector encodes the IDE sense; scorer counts it wrong). Cos-sim analysis pending |
| M2 ablate-final overclaim ("changes nothing", no CI) | Accepted; CIs now computed (54% [47–61] vs 56% [49–63]); language to be softened |
| M3 linear-probe / nearest-vector baselines | TODO |
| M4 single seed, no CIs | Seed 1 training + bootstrap CIs running/done |
| M5 capability retention, transfer eval | MMLU base-vs-adapter running; transfer eval TODO |
| Minor: mentions() substring bugs | Fixed (word-boundary regex) |
| Minor: label-boundary assert in train_lora | TODO |

## Full report

**Recommendation: Reject in current form.**

### FATAL

**F1. "Naming" is plausibly online steering readout, not introspection.**
The injector adds the concept vector at every position of every forward
pass, including each newly generated token. The training target puts the
concept as the final content word of a fixed string. So the LoRA only has
to learn: decode the direction currently sitting in my layer-18 residual
at this position into a token — a learned linear probe wired through the
LM head, not a report about "processing this message." Fixes: (1) prefill
control on the base model — force "YES — … It is about" and measure naming
with injection active; if the untrained model names ~50%, the gain is
format. (2) Inject during prompt only; if naming collapses when injection
stops at generation start, "naming" was never an introspective report.

**F2. The detection/naming dissociation is close to guaranteed by the
intervention geometry.** Detection can never be ablated: the injector
re-adds v at layer 18 at every position/step, so layer 19 always reads the
full perturbation; a residual-norm anomaly detector survives strict
ablation by construction. Naming cannot survive it: fine-grained
directional identity dies when v̂ is zeroed in 9 layers. Any model that
detects via magnitude and names via direction produces exactly the
reported table. Note the internal contradiction: strict ablation DID hurt
detection at strength 2 (84%→54%). Fixes: random-vector injections (if
YES ~100%, detection is a content-agnostic alarm); random-direction
control for the strict ablation; low-norm hard negatives.

**F3. FPR numbers are training-set evaluation.** Clean eval prompts are
byte-for-byte identical to ~120-times-seen training prompts (train loss
0.002 = memorized); mention trials reuse the trained P.S. scaffold.
"Anti-cheat controls hold perfectly" is near-vacuous, and FPR invariance
across ablation conditions is literally vacuous (hooks cleared on
non-injected trials). Fixes: held-out templates, paraphrased + adversarial
clean prompts, n≥100, naturalistic mentions.

**F4. The 3%→54% comparison is confounded by format compliance.** The
base model often answers off-format; scored as misses. Fix: base detection
as AUROC over logprob(YES)−logprob(NO); base naming under prefill.

### MAJOR
M1 generalization narrower than claimed (templates, layer, vector recipe,
extraction prompt all not held out; HELDOUT densely neighbored by TRAIN;
errors land on TRAIN vocabulary; strength 2 is mild extrapolation).
M2 ablate-final rules out less than claimed; 54 vs 56 asserted without CI.
M3 missing trivial baselines: logistic probe on layer-18/19 activations
(detection) and nearest-training-vector classifier (naming) — if these
match the LoRA, the claim shrinks to "the model routes a probe's output
through its own mouth."
M4 statistics: single seed, greedy determinism, tiny clean cells,
unbalanced template assignment; want Wilson CIs, ≥3 seeds, paired tests.
M5 promised-but-missing: capability retention, strength-report task,
LLM-judge scoring, layer holdout, transfer to prompt-level introspection.

### MINOR (abridged)
mentions() substring fragility (fixed); tokenization-boundary assert in
train_lora; per-position strength uncontrolled (typical_norm from
last-token of short prompts); use_cache train/eval mismatch to document;
report per-cell n; one vector family only; soften novelty claim until
controls exist.

### Bar for a workshop paper
Prefill + prompt-only controls; random-vector + random-direction-strict
controls; held-out templates with ≥100 varied clean trials; probe/nearest
baselines; CIs and ≥3 seeds; capability retention; reframed claims — "we
trained a model to verbalize the output of an internal perturbation
detector and a direction decoder" is what the current evidence supports.
