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


def format_table2(rows: dict[str, dict[str, dict]], projects: list[str],
                  show_auc: bool = True) -> str:
    """rows: method -> {project -> metrics}. Renders ACC/F1 columns per project + Combined.

    AUC is appended when available. It is not a paper metric, but accuracy/F1 at one threshold
    hide discriminative power -- which is exactly how a checkpoint-selection bug that reported a
    near-random model went unnoticed.
    """
    cols = projects + ["Combined"]
    width = 22 if show_auc else 16
    header = f"{'Method':<24}" + "".join(f"{c[:16]:>{width}}" for c in cols)
    title = "Table 2: Accuracy / F1" + (" / AUC (%)" if show_auc else " (%)")
    lines = [title, header, "-" * len(header)]
    for method, pm in rows.items():
        cells = ""
        for c in cols:
            m = pm.get(c)
            if m is None:
                cells += f"{'-':>{width}}"
            elif show_auc and m.get("auc") is not None:
                cells += f"{m['accuracy']:6.2f}/{m['f1']:<6.2f}/{m['auc']:<5.2f} "
            else:
                cells += f"{m['accuracy']:6.2f}/{m['f1']:<6.2f}  ".ljust(width)
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
    lines = ["", "Ablation: single-edge vs composite (Combined val, Accuracy / F1 / AUC %)"]
    for model, em in results.items():
        lines.append(f"  [{model}]")
        for edge, m in em.items():
            auc = f"  auc {m['auc']:6.2f}" if m.get("auc") is not None else ""
            lines.append(f"    {edge:<12} acc {m['accuracy']:6.2f}  f1 {m['f1']:6.2f}{auc}")
    return "\n".join(lines)
