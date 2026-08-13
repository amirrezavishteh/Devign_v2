"""Pretty-printers for the paper's result tables."""
from __future__ import annotations

import numpy as np

from training.metrics import binary_metrics


def per_project_eval(probs, labels, projects, threshold=0.5):
    """Returns {project: metrics, 'Combined': metrics} from prob/label/project arrays."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    projects = np.asarray(projects)
    preds = (probs >= threshold).astype(int)
    out = {}
    for proj in sorted(set(projects.tolist())):
        m = projects == proj
        out[proj] = binary_metrics(labels[m], preds[m])
    out["Combined"] = binary_metrics(labels, preds)
    return out


def format_table2(rows: dict[str, dict[str, dict]], projects: list[str]) -> str:
    """rows: method -> {project -> metrics}. Renders ACC/F1 columns per project + Combined."""
    cols = projects + ["Combined"]
    header = f"{'Method':<24}" + "".join(f"{c[:12]:>16}" for c in cols)
    lines = ["Table 2: Accuracy / F1 (%)", header, "-" * len(header)]
    for method, pm in rows.items():
        cells = ""
        for c in cols:
            m = pm.get(c)
            if m is None:
                cells += f"{'-':>16}"
            else:
                cells += f"{m['accuracy']:6.2f}/{m['f1']:<6.2f}  "
        lines.append(f"{method:<24}{cells}")
    return "\n".join(lines)


def format_table3(results: dict, projects: list[str]) -> str:
    cols = projects + ["Combined"]
    header = f"{'Method':<24}" + "".join(f"{c[:12]:>16}" for c in cols)
    lines = ["", "Table 3: Imbalanced setting (10% vuln) Accuracy / F1 (%)", header, "-" * len(header)]
    for method, pm in results.items():
        cells = ""
        for c in cols:
            m = pm.get(c)
            cells += f"{m['accuracy']:6.2f}/{m['f1']:<6.2f}  " if m else f"{'-':>16}"
        lines.append(f"{method:<24}{cells}")
    return "\n".join(lines)


def format_ablation(results: dict[str, dict[str, dict]]) -> str:
    """results: model -> {edge_type/'Composite' -> metrics (combined)}."""
    lines = ["", "Ablation: single-edge vs composite (Combined val, Accuracy / F1 %)"]
    for model, em in results.items():
        lines.append(f"  [{model}]")
        for edge, m in em.items():
            lines.append(f"    {edge:<12} acc {m['accuracy']:6.2f}  f1 {m['f1']:6.2f}")
    return "\n".join(lines)
