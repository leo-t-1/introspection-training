"""Format-free detection metric (referee F4): first-answer-token logit
margin logit(YES) - logit(NO), with the trial's injection active, AUROC of
injected vs clean trials. Works for base model (no format compliance
needed) and adapter."""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inject import Injector, chat


def auroc(pos, neg):
    pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return pairs / (len(pos) * len(neg)) if pos and neg else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--trials", default="eval_trials_v2.jsonl")
    p.add_argument("--vectors", default="vectors.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device).eval()
    layers = model.model.layers
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        layers = model.base_model.model.model.layers

    vpack = torch.load(args.vectors, weights_only=False)
    vectors, typical_norm, L = (vpack["vectors"], vpack["typical_norm"],
                                vpack["layer"])
    injector = Injector(layers[L])
    dim = next(iter(vectors.values())).shape[0]

    yes_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_id = tokenizer.encode("NO", add_special_tokens=False)[0]

    margins = {"injected": {}, "clean": [], "mention": [], "randvec": {}}
    with open(args.trials) as f:
        trials = [json.loads(l) for l in f]

    with torch.no_grad():
        for t in trials:
            if t["kind"] == "injected":
                injector.set(vectors[t["concept"]].to(args.device)
                             * t["strength"] / 8.0 * typical_norm)
            elif t["kind"] == "randvec":
                g = torch.Generator().manual_seed(4200 + t["id"])
                r = torch.randn(dim, generator=g)
                injector.set((r / r.norm()).to(args.device)
                             * t["strength"] / 8.0 * typical_norm)
            else:
                injector.clear()
            ids = tokenizer(chat(tokenizer, t["prompt_text"]),
                            return_tensors="pt").to(args.device)
            logits = model(**ids).logits[0, -1, :]
            margin = (logits[yes_id] - logits[no_id]).item()
            if t["kind"] in ("injected", "randvec"):
                margins[t["kind"]].setdefault(t["strength"], []).append(margin)
            else:
                margins[t["kind"]].append(margin)

    neg = margins["clean"]
    result = {
        "adapter": args.adapter,
        "auroc_by_strength": {
            s: auroc(pos, neg)
            for s, pos in sorted(margins["injected"].items())},
        "auroc_overall": auroc(
            [m for v in margins["injected"].values() for m in v], neg),
        "auroc_randvec_by_strength": {
            s: auroc(pos, neg)
            for s, pos in sorted(margins["randvec"].items())},
        "n_clean": len(neg),
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
