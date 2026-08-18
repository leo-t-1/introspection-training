"""Precompute unit concept vectors for all TRAIN+HELDOUT concepts and the
typical residual norm. Saves vectors.pt = {"vectors": {word: tensor},
"typical_norm": float, "layer": int}."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import concepts
from inject import chat, last_token_hidden


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer-frac", type=float, default=0.66)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="vectors.pt")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device).eval()
    L = int(len(model.model.layers) * args.layer_frac)

    base = torch.stack([
        last_token_hidden(model, tokenizer, f"Tell me about {w}.", L, args.device)
        for w in concepts.BASELINE])
    mean_base = base.mean(0)
    typical_norm = base.norm(dim=-1).mean().item()

    vectors = {}
    for i, c in enumerate(concepts.TRAIN + concepts.HELDOUT):
        v = last_token_hidden(
            model, tokenizer, f"Tell me about {c}.", L, args.device) - mean_base
        vectors[c] = (v / v.norm()).cpu()
        if i % 50 == 0:
            print(f"{i}: {c}")

    torch.save({"vectors": vectors, "typical_norm": typical_norm,
                "layer": L}, args.out)
    print(f"saved {len(vectors)} vectors, layer {L}, "
          f"typical_norm {typical_norm:.1f} -> {args.out}")


if __name__ == "__main__":
    main()
