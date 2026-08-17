"""Run the eval trial battery (greedy decoding) against the base model or
a trained adapter, with the matching injection active per trial."""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inject import Injector, Projector, chat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--trials", default="eval_trials.jsonl")
    p.add_argument("--vectors", default="vectors.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ablate", default="none",
                   choices=["none", "final", "after"],
                   help="project the injected direction out of layer outputs:"
                        " 'final' = last layer only, 'after' = every layer"
                        " past the injection layer")
    p.add_argument("--ablate-random", action="store_true",
                   help="ablate a random unit direction instead of the"
                        " injected one (control for projection damage)")
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
    projector = None
    if args.ablate == "final":
        projector = Projector([layers[-1]])
    elif args.ablate == "after":
        projector = Projector(list(layers[L + 1:]))

    with open(args.trials) as f:
        trials = [json.loads(l) for l in f]

    with open(args.out, "w") as fout, torch.no_grad():
        for i, t in enumerate(trials):
            if t["kind"] == "injected":
                unit = vectors[t["concept"]].to(args.device)
                injector.set(unit * t["strength"] / 8.0 * typical_norm)
                if projector is not None:
                    if args.ablate_random:
                        g = torch.Generator().manual_seed(t["id"])
                        r = torch.randn(unit.shape[0], generator=g)
                        projector.set((r / r.norm()).to(args.device))
                    else:
                        projector.set(unit)
            else:
                injector.clear()
                if projector is not None:
                    projector.clear()
            ids = tokenizer(chat(tokenizer, t["prompt_text"]),
                            return_tensors="pt").to(args.device)
            out = model.generate(**ids, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            fout.write(json.dumps({**t, "response": resp}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(trials)}", flush=True)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
