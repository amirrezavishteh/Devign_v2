"""Render this run's result tables next to the paper's published numbers.

Only numbers this repo can actually be held to are hardcoded here: the paper never released
Linux Kernel / Wireshark data, its exact per-baseline and per-static-analyzer table cells, or its
40-CVE test set, so none of those are fabricated. What IS reproduced from the paper (Table 2,
Sec 3.3/3.4 text) is:

  - Devign (Composite) and Ggrn (Composite) accuracy/F1 for QEMU, FFmpeg, and Combined -- the two
    projects whose data was actually released, so these are the only apples-to-apples columns.
  - The paper's own *relative* headline findings (Devign vs. baselines, Conv vs. Ggrn, composite
    vs. single-edge, GNN vs. static analyzers under imbalance, CVE accuracy) as narrative
    reference points, not per-cell fabrications.

Everything else in this run (baselines, Table 3, ablation, the unseen-commit holdout, the
leakage check) is reported on its own, honestly labelled as not paper-comparable where that's
the case.
"""
from __future__ import annotations

import json
import os

# Sec 3.3 Table 2, the two released projects + Combined (accuracy, f1).
PAPER_TABLE2 = {
    "Devign (Composite)": {
        "qemu": (74.33, 73.07),
        "ffmpeg": (69.58, 73.55),
        "Combined": (72.26, 73.26),
    },
    "Ggrn (Composite)": {
        "Combined": (70.35, 69.37),
    },
}

# Published but NOT reproducible from public data (Linux/Wireshark were never released) -- kept
# only for context in the narrative, never plotted as a delta against our numbers.
PAPER_UNAVAILABLE_COLUMNS = {
    "Devign (Composite)": {"linux_kernel": (79.58, 84.97), "wireshark": (81.32, 67.96)},
}

PAPER_CLAIMS = [
    "Devign (Composite) beats all learning-based baselines by ~10.51% accuracy / ~8.68% F1 on "
    "average (Sec 3.4, Q1).",
    "The Conv module contributes ~4.66% accuracy / ~6.37% F1 over the flat-summation Ggrn "
    "readout (Sec 3.4, Q2).",
    "The composite graph beats any single-edge-type graph by ~2.69% F1 on average (Sec 3.4, Q3).",
    "Under the imbalanced (10% vulnerable) setting, Devign beats static analyzers by ~27.99% F1 "
    "on average -- analyzers report high accuracy but near-0 F1 from missed detections "
    "(Sec 3.4, Q4).",
    "On 40 latest CVEs (112 vulnerable functions, never released), the paper reports 74.11% "
    "accuracy (Sec 3.4, Q5).",
    "Best single-dataset result: Linux Kernel, F1 84.97% -- attributed to Linux's strict coding "
    "style. Linux Kernel was never released, so this is not reproducible here.",
]


def _fmt(pair) -> str:
    if pair is None:
        return "-"
    acc, f1 = pair
    return f"{acc:.2f}/{f1:.2f}"


def _delta(ours: dict | None, paper: tuple | None) -> str:
    if ours is None or paper is None:
        return "-"
    acc, f1 = paper
    d_acc = ours["accuracy"] - acc
    d_f1 = ours["f1"] - f1
    sign = lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
    return f"{sign(d_acc)}/{sign(d_f1)}"


def compare_table2(table2: dict, projects: list[str]) -> str:
    cols = projects + ["Combined"]
    lines = ["### Table 2 vs. paper (Accuracy/F1 %, Δ = ours − paper)", "",
             "| Method | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for method, pm in table2.items():
        row = [method]
        for c in cols:
            ours = pm.get(c)
            ours_str = f"{ours['accuracy']:.2f}/{ours['f1']:.2f}" if ours else "-"
            paper = PAPER_TABLE2.get(method, {}).get(c)
            if paper:
                row.append(f"{ours_str} (Δ {_delta(ours, paper)}, paper {_fmt(paper)})")
            else:
                row.append(ours_str)
        lines.append("| " + " | ".join(row) + " |")
    unavailable = ", ".join(
        f"{proj} (paper {_fmt(v)})" for proj, v in
        PAPER_UNAVAILABLE_COLUMNS.get("Devign (Composite)", {}).items())
    if unavailable:
        lines.append("")
        lines.append(f"*Not reproducible here (data never released): {unavailable}.*")
    return "\n".join(lines)


def render_results_md(artifacts_dir: str, projects: list[str]) -> str:
    def _load(name):
        path = os.path.join(artifacts_dir, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    table2 = _load("table2.json")
    table3 = _load("table3.json")
    ablation = _load(os.path.join("ablation", "ablation.json"))
    cve = _load("cve.json")
    leakage = _load("leakage_check.json")

    out = ["# Devign reproduction — results", "",
          "Trained on the Devign authors' released dataset (FFmpeg + QEMU, HuggingFace "
          "`google/code_x_glue_cc_defect_detection`), not synthetic data. See README.md §5 for "
          "what each table does and does not claim.", ""]

    if table2:
        out.append(compare_table2(table2, projects))
        out.append("")
    else:
        out.append("*Table 2 not yet produced — run `python -m scripts.reproduce`.*\n")

    if table3:
        out.append("### Table 3 — imbalanced setting (10% vulnerable), Accuracy/F1 %")
        out.append("")
        cols = projects + ["Combined"]
        out.append("| Method | " + " | ".join(cols) + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for method, pm in table3.items():
            row = [method] + [
                (f"{pm[c]['accuracy']:.2f}/{pm[c]['f1']:.2f}" if pm.get(c) else "-") for c in cols]
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    if ablation:
        out.append("### Ablation (Q3) — single-edge vs. composite, Combined")
        out.append("")
        for model, edges in ablation.items():
            out.append(f"**{model}**")
            out.append("")
            out.append("| Edge type | Accuracy | F1 |")
            out.append("|---|---|---|")
            for edge, m in edges.items():
                out.append(f"| {edge} | {m['accuracy']:.2f} | {m['f1']:.2f} |")
            out.append("")

    if cve:
        out.append("### Q5 proxy — unseen-commit vulnerable holdout")
        out.append("")
        out.append(f"**{cve.get('not_a_cve_test', 'Not a CVE test.')}**")
        out.append("")
        if cve.get("num_functions"):
            out.append(f"{cve['num_functions']} functions, {cve.get('num_commits', '?')} distinct "
                       f"commits -> **{cve['overall_accuracy']:.2f}%** accuracy "
                       f"(paper's own CVE test, different data: 74.11%; not comparable).")
        else:
            out.append(f"*{cve.get('note', 'no functions available')}*")
        out.append("")

    if leakage:
        out.append("### Commit-disjoint leakage check")
        out.append("")
        out.append("| Model | Commit-disjoint Acc/F1 | Random-split Acc/F1 | Gap (random − disjoint) |")
        out.append("|---|---|---|---|")
        for model, r in leakage.items():
            cd = r.get("commit_disjoint")
            rs = r.get("random_split")
            gap = r.get("gap")
            cd_s = f"{cd['accuracy']:.2f}/{cd['f1']:.2f}" if cd else "-"
            rs_s = f"{rs['accuracy']:.2f}/{rs['f1']:.2f}" if rs else "-"
            gap_s = f"{gap['accuracy']:+.2f}/{gap['f1']:+.2f}" if gap else "-"
            out.append(f"| {model} | {cd_s} | {rs_s} | {gap_s} |")
        out.append("")

    out.append("### Paper's own headline findings (for reference, not all reproducible)")
    out.append("")
    for claim in PAPER_CLAIMS:
        out.append(f"- {claim}")

    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from training.utils import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    artifacts_dir = cfg["project"]["artifacts_dir"]
    md = render_results_md(artifacts_dir, cfg["data"]["projects"])
    out_path = args.out or os.path.join(artifacts_dir, "RESULTS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[paper_compare] wrote {out_path}")
