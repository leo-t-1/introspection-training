# Training LLM Introspection

**Can you fine-tune a language model to accurately report on its own
internal states — and if so, what did it actually learn?**

This project injects "thoughts" (concept vectors) into a 7B model's residual
stream, following Anthropic's [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html)
protocol, then LoRA-trains the model to detect and name them. It goes beyond
prior trainability results by asking the harder question: *is the trained
ability a genuine self-report, or a probe wired through the model's mouth?*
An adversarial red-team review ([REVIEW.md](REVIEW.md)) and the control
experiments it demanded are part of the repo.

## Headline results (Qwen2.5-7B, 50 held-out concepts)

| Metric | Base | Trained (LoRA) |
|---|---|---|
| Detect + correctly name injected concept | 3% | **54%** (seed 1: 59%) |
| Detection TPR, strengths 4–8 | 20–50% | **100%** |
| Detection TPR, strength 2 (below training band) | 4% | **84%** |
| False positives, adversarial clean prompts (never-trained templates) | — | **0/54** |
| MMLU (capability retention) | 67.75% | 67.5% |

## What the controls showed

1. **Naming is a genuine trained report, not format.** The untrained model
   given the exact answer template names the injected concept 3% of the
   time; and naming survives (44%) when injection is *cut off at the first
   generated token* — the model reports what happened while it read the
   prompt, via attention, not a live signal under the output head.
2. **Detection is a verbalized anomaly alarm.** Random directions at
   matched norm trigger "YES" 100% at high strength, and the *base* model
   already separates injected from clean trials at AUROC 0.82–0.98 in its
   logits. Training mainly makes a pre-existing perturbation signal
   speakable. Near threshold, detection becomes direction-sensitive.
3. **Detection and naming dissociate mechanistically.** Projecting the
   injected direction out of every post-injection layer collapses naming
   (54%→8%) but leaves detection at 100%; projecting it out at the final
   layer only changes nothing — so naming is decoded from mid-layer
   residuals, not from the output distribution. Random-direction
   projections do no damage (controls).
4. **Error structure:** wrong names are semantic neighbors (balloon→kite,
   cheese→milk); some correct readouts fall outside every training
   vocabulary (seashell→"pearl"; the "eclipse" vector — built from
   "Tell me about eclipse." — is reported as *editor/debugger*, i.e. the
   IDE sense the vector actually encodes).

Full numbers, caveats and the red-team verdict: [RESULTS.md](RESULTS.md) ·
[REVIEW.md](REVIEW.md) · plan: [PLAN.md](PLAN.md)

## Reproduce

```bash
pip install -r requirements.txt          # needs one ~80GB GPU (A100)
python gen_trials.py                     # 3000 train + 235 eval trials
python gen_vectors.py                    # concept vectors @ layer 18
python eval_intro.py --trials eval_trials.jsonl --out results/eval_base.jsonl
python train_lora.py --out adapters/introspect-lora        # ~1h on A100
python eval_intro.py --adapter adapters/introspect-lora \
    --trials eval_trials.jsonl --out results/eval_trained.jsonl
python score.py results/eval_trained.jsonl
```

Key flags: `--ablate {final,after} [--ablate-random]` (ablation probes),
`--inject-prompt-only` (report-vs-live-readout control), `--prefill`
(format control), `gen_eval_v2.py` (held-out templates, adversarial cleans,
random vectors). `boot_ci.py` gives bootstrap CIs; `cap_eval.py` MMLU;
`auroc_base.py` format-free detection.

The trained adapter (154MB) is on the
[v0.1-adapter release](../../releases/tag/v0.1-adapter).

## Relation to prior work

- [Lindsey 2026](https://arxiv.org/abs/2601.01828): emergent (zero-shot)
  introspective awareness in frontier models — the protocol we borrow.
- [Steering Awareness](https://arxiv.org/abs/2511.21399) and
  [IFT](https://arxiv.org/abs/2607.14111): showed injection-detection is
  trainable. This repo's delta: the report-vs-readout controls, the
  detection/naming dissociation via directional ablation, and the
  anomaly-alarm characterization of detection.

## Honest limitations

Trained ≠ emergent: this is installed, not discovered, self-access. One
model family, one vector recipe, two seeds. Naming has a modest
live-readout component at low strength. 2/25 naturalistic mention decoys
drew a false YES. String-match scoring makes identification a lower bound.
Linear-probe baselines and transfer to prompt-level introspection
(self-prediction, calibration) are open.

## License

MIT — see [LICENSE](LICENSE).
