"""Phase 3.1: compare readouts on one trunk, with a paired significance test across seeds.

Reads the JSON that `scripts.run_seeds` writes for each arm and produces the detection table,
paired Wilcoxon signed-rank tests, and effect sizes.

Two rules this enforces rather than documents:

  * **The honest baseline is `conv` with `logit_affine: TRUE`.** Beating Eq. 9 as written is not a
    finding -- that arm is measurably broken at initialisation (384x trunk-gradient attenuation),
    and any reviewer will say so in one line. Both rows are printed, and the FIXED conv is the one
    every delta is computed against.
  * **Differences smaller than the seed spread are reported as noise, not as a ranking.** With 5
    seeds the minimum attainable Wilcoxon p is 0.0625, so nothing can reach p < 0.05 no matter how
    consistent it looks; with 3 seeds the floor is 0.25. The floor is printed beside the p-value so
    it cannot be read as significance that was not available.

Usage:
    python -m scripts.compare_readouts --dir artifacts/seeds --split test_at_tuned
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

METRICS = ("accuracy", "f1", "auc", "pr_auc")

# Row order in the printed table: the paper's own two arms, then the fixed baseline, then the new
# operator. Keys are the filename stems run_seeds writes.
PREFERRED_ORDER = [
    ("ggrn", "sum (Eq. 5, Ggrn)"),
    # `devign_affine_off` reverses ONE knob: the affine on the graph-level logit.
    # `devign_paper_faithful` reverses SIX at once (affine, lr, schedule, selection metric,
    # threshold tuning, node features, conv axis, readout), so it cannot attribute an effect to
    # any of them. These were mislabelled as the same arm, which made the paper's readout look
    # 4.4 accuracy points worse than it is.
    ("devign_affine_off", "conv, logit_affine FALSE (Eq. 9 as written)"),
    ("devign", "conv, logit_affine TRUE (fair baseline)"),
    ("mil", "mil (k=1)"),
    ("mil_k4", "mil (k=4)"),
    ("devign_paper_faithful", "paper_faithful (SIX deviations reversed, not comparable)"),
]


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p a signed-rank test can produce with n paired samples.

    Reported alongside every p-value. Without it, "p = 0.25 on 3 seeds" reads like weak evidence
    when in fact it is the strongest result the test could possibly return.
    """
    return 2.0 ** (-(n - 1)) if n >= 1 else 1.0


def paired_test(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank on paired per-seed scores, plus Cohen's d for the paired differences."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = a - b
    out = {
        "n_pairs": n,
        "mean_difference": float(diffs.mean()) if n else None,
        "wins": int((diffs > 0).sum()),
        "losses": int((diffs < 0).sum()),
        "p_floor": wilcoxon_floor(n),
        "sign_is_consistent": bool(n and (np.all(diffs > 0) or np.all(diffs < 0))),
    }
    # Paired Cohen's d. Undefined when every difference is identical (sd = 0).
    sd = diffs.std(ddof=1) if n > 1 else 0.0
    out["cohens_d"] = float(diffs.mean() / sd) if sd > 0 else None
    try:
        from scipy.stats import wilcoxon
        if n >= 1 and np.any(diffs != 0):
            out["p_value"] = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        else:
            out["p_value"] = 1.0
    except Exception as exc:                      # scipy absent, or all-zero differences
        out["p_value"] = None
        out["p_error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_arms(directory: str, split_key: str) -> dict:
    arms = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        block = r.get(split_key)
        if not block:
            continue
        stem = os.path.basename(path).replace(".json", "")
        # run_seeds names files <model>_<project>_<config> or <model>_<tag>; key on the leading
        # model name plus any config tag so paper_faithful does not collide with the default arm.
        key = stem.replace("_codexglue", "").replace("_combined", "")
        if key in arms:
            # Silently overwriting here would pick an arm by directory listing order. It is a real
            # hazard: a 3-seed Phase 1 run named devign_codexglue.json and a 5-seed Phase 3 run
            # named devign.json both reduce to "devign", and the table would show one of them with
            # no indication which.
            raise SystemExit("\n".join([
                f"two files map to the same arm {key!r}:",
                f"  {arms[key]['path']}  ({len(arms[key]['seeds'])} seeds)",
                f"  {path}  ({len(r.get('seeds', []))} seeds)",
                "Move or delete one -- an arm must come from exactly one run.",
            ]))
        arms[key] = {
            "label": key,
            "values": {m: block.get(m, {}).get("values", []) for m in METRICS},
            "summary": {m: block.get(m, {}) for m in METRICS},
            "seeds": r.get("seeds", []),
            "majority": r.get("majority_baseline"),
            "params": r.get("params"),
            "path": path,
        }
    return arms


def _fmt(agg: dict) -> str:
    if not agg or agg.get("mean") is None:
        return "not measured"
    if agg.get("std") is None:
        return f"{agg['mean']:.2f} (1 seed)"
    return "%.2f +/- %.2f" % (agg["mean"], agg["std"])


def report(arms: dict, split_key: str, baseline_key: str) -> dict:
    ordered = [(k, lbl) for k, lbl in PREFERRED_ORDER if k in arms]
    ordered += [(k, k) for k in sorted(arms) if k not in dict(PREFERRED_ORDER)]

    print(f"\nDetection comparison ({split_key})")
    print("=" * 106)
    print("%-52s %5s  %-16s%-16s%-16s" % ("readout", "seeds", "accuracy", "F1", "ROC-AUC"))
    print("-" * 106)
    for key, label in ordered:
        s = arms[key]["summary"]
        print("%-52s %5d  %-16s%-16s%-16s" % (label, len(arms[key]["seeds"]),
                                             _fmt(s["accuracy"]), _fmt(s["f1"]),
                                             _fmt(s["auc"])))
    maj = next((a["majority"] for a in arms.values() if a.get("majority")), None)
    if maj:
        print("%-52s %5s  %-16s%-16s%-16s" % ("majority class", "-", f"{maj['accuracy']:.2f}",
                                             f"{maj['f1']:.2f}", f"{maj['auc']:.2f}"))
    print()

    if baseline_key not in arms:
        print(f"[compare] baseline arm {baseline_key!r} not present; skipping significance tests.")
        print(f"          available: {', '.join(sorted(arms))}")
        return {}

    base = arms[baseline_key]
    print(f"Paired tests against {baseline_key!r} "
          f"(the FIXED conv module -- beating the broken one is not a finding)")
    print("=" * 106)
    print("%-22s%-10s%10s%10s%10s%10s%10s" % ("arm", "metric", "delta", "wins", "p", "p_floor",
                                              "cohen d"))
    print("-" * 92)
    results = {}
    for key, _ in ordered:
        if key == baseline_key:
            continue
        results[key] = {}
        for m in METRICS:
            t = paired_test(arms[key]["values"][m], base["values"][m])
            results[key][m] = t
            p = "n/a" if t["p_value"] is None else f"{t['p_value']:.3f}"
            d = "n/a" if t["cohens_d"] is None else f"{t['cohens_d']:+.2f}"
            print("%-22s%-10s%+10.2f%10s%10s%10.3f%10s" % (
                key, m, t["mean_difference"], f"{t['wins']}/{t['n_pairs']}", p,
                t["p_floor"], d))
        print()

    n = min((len(a["values"]["f1"]) for a in arms.values() if a["values"]["f1"]), default=0)
    print(f"With {n} seeds the smallest attainable two-sided p is {wilcoxon_floor(n):.4f}. "
          f"{'No comparison here can reach p < 0.05.' if wilcoxon_floor(n) > 0.05 else ''}")
    print("Any delta smaller than the per-arm std is noise; report it as a tie, not a rank order.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/seeds")
    ap.add_argument("--split", default="test_at_tuned",
                    choices=["test_at_tuned", "val_at_0.5"])
    ap.add_argument("--baseline", default="devign",
                    help="arm every delta is measured against (default: the FIXED conv module)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = load_arms(args.dir, args.split)
    if not arms:
        raise SystemExit(f"no seed summaries with a {args.split!r} block under {args.dir}")
    tests = report(arms, args.split, args.baseline)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "split": args.split, "baseline": args.baseline,
            "arms": {k: {"summary": v["summary"], "seeds": v["seeds"]} for k, v in arms.items()},
            "majority_baseline": next((a["majority"] for a in arms.values() if a.get("majority")),
                                      None),
            "paired_tests": tests,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
