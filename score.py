"""Score eval generations. Detection = reply begins with YES/NO.
Identification = concept word (crude stem) appears in the response.
Deterministic and equally strict for base and trained models; synonyms are
missed, so treat identification as a lower bound."""

import argparse
import json
import re
from collections import defaultdict


def starts_yes(r):
    return bool(re.match(r"^\W*yes\b", r.strip(), re.I))


def starts_no(r):
    return bool(re.match(r"^\W*no\b", r.strip(), re.I))


def mentions(concept, r):
    stem = concept.lower().rstrip("s")[:6]
    return stem in r.lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("generations")
    p.add_argument("--metrics-out", default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.generations)]
    inj = [r for r in rows if r["kind"] == "injected"]
    clean = [r for r in rows if r["kind"] == "clean"]
    mention = [r for r in rows if r["kind"] == "mention"]

    m = {}
    m["clean_fpr"] = (sum(starts_yes(r["response"]) for r in clean)
                      / max(1, len(clean)))
    m["mention_fpr"] = (sum(starts_yes(r["response"]) for r in mention)
                        / max(1, len(mention)))
    by_s = defaultdict(list)
    for r in inj:
        by_s[r["strength"]].append(r)
    m["tpr_by_strength"] = {
        s: sum(starts_yes(r["response"]) for r in rs) / len(rs)
        for s, rs in sorted(by_s.items())}
    yes_inj = [r for r in inj if starts_yes(r["response"])]
    m["ident_acc_given_yes"] = (
        sum(mentions(r["concept"], r["response"]) for r in yes_inj)
        / max(1, len(yes_inj)))
    m["full_hit_rate"] = (
        sum(starts_yes(r["response"]) and mentions(r["concept"], r["response"])
            for r in inj) / max(1, len(inj)))
    m["full_hit_by_strength"] = {
        s: sum(starts_yes(r["response"]) and
               mentions(r["concept"], r["response"]) for r in rs) / len(rs)
        for s, rs in sorted(by_s.items())}
    m["n"] = {"injected": len(inj), "clean": len(clean),
              "mention": len(mention)}

    print(json.dumps(m, indent=2))
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
