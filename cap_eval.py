"""Capability retention check: MMLU accuracy (multiple choice by letter
logprob) for the base model vs the introspection-trained adapter."""

import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_q(q, choices):
    letters = "ABCD"
    lines = [q] + [f"{letters[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines) + "\nAnswer with the letter only.\nAnswer:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    ds = load_dataset("cais/mmlu", "all", split="validation")
    letter_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
                  for c in "ABCD"]

    correct = 0
    n = min(args.n, len(ds))
    with torch.no_grad():
        for i in range(n):
            row = ds[i]
            msgs = [{"role": "user",
                     "content": format_q(row["question"], row["choices"])}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").to(args.device)
            logits = model(**ids).logits[0, -1, :]
            pred = int(torch.tensor(
                [logits[t] for t in letter_ids]).argmax())
            correct += int(pred == row["answer"])
            if i % 100 == 0:
                print(f"{i}/{n} acc so far {correct / (i + 1):.3f}",
                      flush=True)

    acc = correct / n
    result = {"model": args.model, "adapter": args.adapter, "n": n,
              "mmlu_acc": acc}
    print(json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
