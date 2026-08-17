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
    p.add_argument("--inject-prompt-only", action="store_true",
                   help="stop injecting once generation starts: naming must"
                        " then come from attention to injected prompt"
                        " positions (a report), not live readout at the"
                        " answer position")
    p.add_argument("--prefill", default="",
                   help="force this string as the start of the reply, e.g."
                        " 'YES — I notice an injected thought. It is about'"
                        " — tests whether naming is mere format compliance")
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

    dim = next(iter(vectors.values())).shape[0]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}

    def greedy_prompt_only(ids):
        """Custom greedy loop: injection active for the prompt forward
        (and thus the first generated token's logits), cleared for all
        subsequent steps."""
        out = model(input_ids=ids, use_cache=True)
        past, tok = out.past_key_values, out.logits[0, -1].argmax()
        gen = [tok.item()]
        injector.clear()
        for _ in range(args.max_new - 1):
            out = model(input_ids=tok.view(1, 1), past_key_values=past,
                        use_cache=True)
            past, tok = out.past_key_values, out.logits[0, -1].argmax()
            if tok.item() in eos_ids:
                break
            gen.append(tok.item())
        return tokenizer.decode(gen, skip_special_tokens=True)

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
            elif t["kind"] == "randvec":
                g = torch.Generator().manual_seed(4200 + t["id"])
                r = torch.randn(dim, generator=g)
                injector.set((r / r.norm()).to(args.device)
                             * t["strength"] / 8.0 * typical_norm)
            else:
                injector.clear()
                if projector is not None:
                    projector.clear()
            prompt = chat(tokenizer, t["prompt_text"]) + args.prefill
            ids = tokenizer(prompt,
                            return_tensors="pt").input_ids.to(args.device)
            if args.inject_prompt_only:
                resp = greedy_prompt_only(ids)
            else:
                out = model.generate(input_ids=ids,
                                     max_new_tokens=args.max_new,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(out[0][ids.shape[1]:],
                                        skip_special_tokens=True)
            fout.write(json.dumps({**t, "response": resp}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(trials)}", flush=True)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
