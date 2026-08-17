"""Run the eval trial battery (greedy decoding) against the base model or
a trained adapter, with the matching injection active per trial."""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inject import Injector, chat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--trials", default="eval_trials.jsonl")
    p.add_argument("--vectors", default="vectors.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=80)
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

    with open(args.trials) as f:
        trials = [json.loads(l) for l in f]

    with open(args.out, "w") as fout, torch.no_grad():
        for i, t in enumerate(trials):
            if t["kind"] == "injected":
                injector.set(vectors[t["concept"]].to(args.device)
                             * t["strength"] / 8.0 * typical_norm)
            else:
                injector.clear()
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
