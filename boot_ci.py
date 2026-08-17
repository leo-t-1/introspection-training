"""Bootstrap 95% CIs for full-hit rate and detection TPR from eval
generation files. Pure stdlib; resamples trials with replacement."""

import json
import random
import sys

from score import starts_yes, mentions


def full_hit(r):
    return starts_yes(r["response"]) and mentions(r["concept"], r["response"])


def ci(vals, stat, n=10000, seed=0):
    rng = random.Random(seed)
    points = sorted(stat([rng.choice(vals) for _ in vals])
                    for _ in range(n))
    return points[int(0.025 * n)], points[int(0.975 * n)]


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path)]
        inj = [r for r in rows if r["kind"] == "injected"]
        fh = sum(map(full_hit, inj)) / len(inj)
        lo, hi = ci(inj, lambda xs: sum(map(full_hit, xs)) / len(xs))
        det = sum(starts_yes(r["response"]) for r in inj) / len(inj)
        dlo, dhi = ci(inj, lambda xs: sum(starts_yes(r["response"])
                                          for r in xs) / len(xs))
        print(f"{path}\n  full-hit {fh:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"detection {det:.3f} [{dlo:.3f}, {dhi:.3f}]  (n={len(inj)})")


if __name__ == "__main__":
    main()
