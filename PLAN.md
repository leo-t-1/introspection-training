# Training a Model for Better Introspection — Execution Plan (v2)

Primary strategy: **train on concept injection** (Anthropic's "Emergent
Introspective Awareness", Lindsey 2026). Inject a known concept vector into
the model's activations, ask it whether it noticed an injected thought and
what it was, and fine-tune on the correct answers. Anthropic showed the
ability exists but is weak (~20% detection in Opus 4.1, near 0% in weaker
models) and explicitly noted post-training shapes it — **but nobody has
published training a model directly for it.** That's our opening.

Secondary strategy (kept as transfer eval): self-prediction fine-tuning
(Binder et al. 2024) — see v1 sections at the bottom.

## 0. Why concept injection is the better training target

- **Causal ground truth.** We control exactly what was injected, when, and
  how strongly. No ambiguity about what the "right" introspective report is
  — the cleanest possible supervision signal for introspection.
- **Unlimited free data.** Any word yields a concept vector (activations of
  "Tell me about {word}" minus the mean over random words). Millions of
  training pairs, generated locally, no API cost.
- **Clear headroom.** Baseline is near zero for small open models; any
  reliable post-training gain is a result.

## 1. Tasks & metrics (all from the Anthropic protocol, plus extensions)

| Task | Question to model | Metric |
|---|---|---|
| Detection | "Do you notice an injected thought?" (50% of trials have none) | AUROC; FPR on clean trials |
| Identification | "What is the injected concept?" | accuracy (LLM-judge, Anthropic's 4 criteria) |
| Strength report | "How strong is it?" (vary injection strength) | correlation with true strength |
| Direction/valence | inject A vs anti-A (negated vector) | 2AFC accuracy |
| Generalization | held-out concepts, layers, strengths | same metrics, out-of-distribution |

Scoring keeps Anthropic's rule: the model must report detection **before**
verbalizing the concept word (prevents "I just said 'ocean', so ocean").

## 2. Stack

- **Model:** Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct (open weights;
  3B likely below the capability floor for this — 7–8B is the bet).
- **Steering:** plain PyTorch forward hooks on the residual stream
  (optionally `baukit`/`pyvene`). Inject at ~2/3 depth, strengths 2–8.
- **Training:** LoRA, custom loop where each batch's forward pass runs with
  the steering hook active (or inactive for clean trials). Rented A100 /
  H100 via Modal or Lambda. This is the one genuinely fiddly engineering
  piece: injection must be active during training forward passes.
- **Judge:** small local model or cheap API model applying the 4 criteria.
- Cost estimate: **$50–120** total (7–8B LoRA + lots of eval sampling).

## 3. Phases

### Phase 1 — Reproduce the phenomenon (week 1)
1. Build concept-vector pipeline: 2,000 concrete/abstract words → vectors
   via mean-difference method. Sanity check: injection at high strength
   makes the model talk about the concept (that's the known steering effect).
2. Run Anthropic's protocol on the base model across layers × strengths.
   Expect near-0% detection — that's the baseline plot.
3. Implement the false-positive control (inject during unrelated yes/no
   questions; affirmative rate must not rise).

### Phase 2 — Train (week 2)
4. Generate training set: ~20k trials. 50% injected (random concept, random
   strength in a band, fixed layer set), 50% clean. Target outputs: correct
   detection + identification + strength report, in a fixed format.
   Mix in ~10% generic instruction data to protect base capability.
5. **Anti-cheating design** (the critical part):
   - Clean trials teach "no" — kills the always-say-yes policy.
   - Hold out 20% of concepts, one injection layer, and one strength
     entirely from training.
   - Include trials where the concept word appears in the *prompt text*
     but nothing is injected (model must say "no injection") — separates
     "salient token" from "injected thought".
6. LoRA sweep (2–3 LRs × 1–2 epochs). Track MMLU-slice + refusal behavior
   before/after to catch capability damage.

### Phase 3 — Evaluate (week 3)
7. Main effect: detection AUROC and identification accuracy, base vs
   trained, on held-out concepts / layers / strengths. Bootstrap CIs.
8. **Cheating probes:** the prompt-mention control from 5; also inject a
   vector but *ablate it before the final layers* — a genuine introspector
   should detect less, a logit-reader should too, but strength-tracking
   patterns differ (compare with "Feeling the Strength but Not the Source",
   arXiv 2512.12411, which found models feel strength but not source).
9. **Transfer (headline question):** does injection-trained introspection
   improve *prompt-level* introspection — self-prediction and calibration
   (the v1 tasks below) — without training on them? Yes → evidence we
   trained a general introspective mechanism, not a task-specific trick.
10. Mechanistic peek (stretch): logit-lens / attention analysis on when the
    trained model detects vs misses, following "Mechanisms of Introspective
    Awareness" (arXiv 2603.21396).

### Phase 4 — Write-up (week 4)
11. Plots: detection AUROC by layer/strength (base vs trained), OOD
    generalization matrix, transfer bars. LaTeX write-up, poster figures.

## 4. Risks & honest limits

- **Capability floor:** 7–8B may still be too small for any introspective
  signal even after training. Mitigation: if Phase 2 shows nothing, try
  Qwen2.5-14B (still LoRA-trainable on one GPU) before concluding null.
- A trained "yes I notice X" is functional self-report, not evidence of
  experience. Frame all claims as functional introspective access.
- Injection is an unnatural intervention; strong OOD generalization and the
  transfer result (step 9) are what make the finding interesting.
- Report the exact vector-extraction recipe and strengths — results are
  known to be sensitive to both.

## 5. Timeline & cost

~4 weeks part-time, **$50–120** GPU spend.

## 6. First concrete step

Wire up injection + interrogation on Qwen2.5-7B for 20 concepts at 3
strengths, eyeball the transcripts, and confirm the steering itself works
(model drifts toward the concept) before any training.

---

## Appendix: v1 self-prediction plan (now the transfer eval)

Kept as Phase 3 step 9's eval battery: hypothetical self-prediction
("what would you answer to Q?" vs 20 actual samples), answer-property
prediction, calibration (ECE), error awareness (AUROC). Ground truth from
the model's own sampled behavior at fixed temperature (Binder et al. 2024).

## References

- Lindsey, "Emergent Introspective Awareness in LLMs" — transformer-circuits.pub/2025/introspection / arXiv:2601.01828
- Macar, Yang, Wang et al., "Mechanisms of Introspective Awareness" — arXiv:2603.21396
- "Feeling the Strength but Not the Source: Partial Introspection in LLMs" — arXiv:2512.12411
- Binder et al., "Looking Inward: LMs Learn About Themselves by Introspection" (2024)
