"""LoRA fine-tuning for introspection: teach the model to detect and
identify concept vectors injected into its own residual stream.

Each training example runs its forward pass with the example's injection
hook active (or inactive for clean/mention trials), and the loss is
computed only on the target tokens. Batch size 1 + grad accumulation so
every example can carry a different injection.
"""

import argparse
import json
import random
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from inject import Injector, chat


def load_trials(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--trials", default="train_trials.jsonl")
    p.add_argument("--vectors", default="vectors.pt")
    p.add_argument("--out", default="adapters/introspect-lora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="use first N trials")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device)
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    vpack = torch.load(args.vectors, weights_only=False)
    vectors, typical_norm, L = (vpack["vectors"], vpack["typical_norm"],
                                vpack["layer"])
    # hook goes on the base model's decoder layer inside the peft wrapper
    injector = Injector(model.base_model.model.model.layers[L])

    trials = load_trials(args.trials)
    random.Random(args.seed).shuffle(trials)
    if args.limit:
        trials = trials[:args.limit]

    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=args.lr)
    model.train()

    step, running = 0, 0.0
    for epoch in range(args.epochs):
        for t in trials:
            prompt = chat(tokenizer, t["prompt_text"])
            p_ids = tokenizer(prompt, return_tensors="pt").input_ids
            full = tokenizer(prompt + t["target"] + tokenizer.eos_token,
                             return_tensors="pt").input_ids
            labels = full.clone()
            labels[0, :p_ids.shape[1]] = -100
            full, labels = full.to(args.device), labels.to(args.device)

            if t["kind"] == "injected":
                vec = (vectors[t["concept"]].to(args.device)
                       * t["strength"] / 8.0 * typical_norm)
                injector.set(vec)
            else:
                injector.clear()

            loss = model(input_ids=full, labels=labels).loss / args.accum
            loss.backward()
            running += loss.item()
            step += 1
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(trials)} "
                      f"loss {running / 200 * args.accum:.4f}", flush=True)
                running = 0.0

    injector.clear()
    model.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
