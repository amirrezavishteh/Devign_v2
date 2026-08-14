"""End-to-end reproduction of the paper's study on real data.

    prepare -> train Devign & Ggrn (per-project + Combined) -> train baselines (same matrix)
            -> assemble Table 2 -> imbalanced Table 3 -> unseen-commit holdout (Q5)
            -> single-edge ablation (Q3, Combined) -> commit-disjoint leakage check

Every stage writes its own artifact and is SKIPPED on a re-run if that artifact already exists,
so an interrupted overnight run resumes instead of restarting from scratch. Pass --force to redo
everything.

Usage:
    python -m scripts.reproduce --config config.yaml
    python -m scripts.reproduce --quick                 # 8-epoch smoke run
    python -m scripts.reproduce --epochs 100             # overnight schedule (see README)
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

from data.prepare import prepare
from evaluation.report import format_table2, format_table3
from scripts.run_ablation import ARTIFACT_PATH as ABLATION_ARTIFACT_PATH
from scripts.run_ablation import run_ablation
from scripts.train import artifact_dir, save_graph_model, train_graph_model
from training.utils import ensure_dir, load_config, resolve_device, set_seed

GRAPH_MODELS = ["devign", "ggrn"]
BASELINE_LABELS = {
    "bilstm": "3-layer BiLSTM",
    "bilstm_att": "3-layer BiLSTM + Att",
    "cnn": "CNN",
    "xgboost": "Metrics + XGBoost",
}


def _stage_done(path: str, force: bool) -> bool:
    return (not force) and os.path.exists(path)


def _matrix_projects(cfg) -> list[str]:
    return list(cfg["data"]["projects"]) + ["combined"]


def _run_graph_matrix(cfg, device, epochs, force, verbose=True):
    """Train Devign + Ggrn on every project + Combined. Returns {model: {project: metrics}}."""
    out = {name: {} for name in GRAPH_MODELS}
    for name in GRAPH_MODELS:
        for project in _matrix_projects(cfg):
            metrics_path = os.path.join(artifact_dir(cfg, name, project), "metrics.json")
            if _stage_done(metrics_path, force):
                with open(metrics_path) as f:
                    out[name][project] = json.load(f)
                print(f"[reproduce] {name}/{project}: skipped (already trained)")
                continue
            print(f"[reproduce] training {name}/{project}")
            model, metrics, train_ds = train_graph_model(
                cfg, name, device, epochs=epochs, project=project, verbose=verbose)
            save_graph_model(cfg, model, name, project, metrics, train_ds.type_vocab_size)
            out[name][project] = metrics
    return out


def _run_baseline_matrix(cfg, device, force, verbose=True):
    """Same project x Combined matrix for the 4 Table-2 baselines."""
    from training.train_baselines import train_bilstm, train_cnn, train_metrics_xgboost

    out = {k: {} for k in BASELINE_LABELS}
    failures = []
    for project in _matrix_projects(cfg):
        out_dir = ensure_dir(os.path.join(cfg["project"]["artifacts_dir"], "baselines", project))

        def _run(key, fn):
            metrics_path = os.path.join(out_dir, f"{key}.json")
            if _stage_done(metrics_path, force):
                with open(metrics_path) as f:
                    out[key][project] = json.load(f)
                print(f"[reproduce] baseline {key}/{project}: skipped (already trained)")
                return
            print(f"[reproduce] training baseline {key}/{project}")
            try:
                metrics = fn()
                out[key][project] = metrics
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as exc:  # a broken baseline must not silently drop the others
                print(f"[reproduce] FAILED: baseline {key}/{project}: {exc}")
                traceback.print_exc()
                failures.append(f"{key}/{project}")

        _run("bilstm", lambda: train_bilstm(cfg, device, attention=False, verbose=verbose,
                                            project=project)[1])
        _run("bilstm_att", lambda: train_bilstm(cfg, device, attention=True, verbose=verbose,
                                                project=project)[1])
        _run("cnn", lambda: train_cnn(cfg, device, verbose=verbose, project=project)[1])
        _run("xgboost", lambda: train_metrics_xgboost(cfg, verbose=verbose, project=project)[1])

    return out, failures


def _relabel_combined(per_project: dict) -> dict:
    """The matrix trains on the lowercase 'combined' project key; report tables (format_table2,
    per_project_eval) all key the aggregate column as 'Combined'."""
    out = dict(per_project)
    if "combined" in out:
        out["Combined"] = out.pop("combined")
    return out


def _assemble_table2(cfg, graph_results, baseline_results, use_test: bool):
    """{method: {project: metrics}} using best_val, or the held-out test split if present."""
    key = "test" if use_test else "best_val"

    def _pick(metrics_dict):
        return metrics_dict.get(key) or metrics_dict.get("best_val")

    table2 = {}
    table2["Devign (Composite)"] = _relabel_combined(
        {p: _pick(m) for p, m in graph_results["devign"].items()})
    table2["Ggrn (Composite)"] = _relabel_combined(
        {p: _pick(m) for p, m in graph_results["ggrn"].items()})
    for key_name, label in BASELINE_LABELS.items():
        table2[label] = _relabel_combined(dict(baseline_results.get(key_name, {})))
    return table2


def _run_imbalanced(cfg, device):
    import pickle

    from data.word2vec_embed import NodeFeaturizer
    from evaluation.imbalanced import (evaluate_devign_imbalanced,
                                       evaluate_static_analyzers, make_imbalanced)
    from models.devign import build_model

    proc = cfg["data"]["processed_dir"]
    with open(os.path.join(proc, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    source_split = splits.get("test") or splits["val"]
    imb = make_imbalanced(source_split, cfg["data"]["imbalanced_vuln_ratio"],
                          cfg["project"]["seed"])

    results = {}
    sa = evaluate_static_analyzers(imb, cfg["evaluation"].get("static_analyzer_paths"))
    for label, r in sa.items():
        results[label] = r["metrics"]

    model_path = os.path.join(artifact_dir(cfg, "devign", "combined"), "model.pt")
    if os.path.exists(model_path):
        import torch

        from data.graph_builder import EDGE_TYPES

        featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
        model = build_model("devign", cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                            type_vocab_size=len(featurizer.type_vocab),
                            num_edge_types=len(EDGE_TYPES))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        metrics, thr = evaluate_devign_imbalanced(cfg, model, device, splits, featurizer,
                                                 EDGE_TYPES)
        if metrics:
            results["Devign (Composite)"] = metrics
            print(f"[reproduce] Table 3 Devign threshold recalibrated to "
                  f"{thr:.3f} for {cfg['data']['imbalanced_vuln_ratio']:.0%} prevalence")
    else:
        print("[reproduce] no Combined Devign model found; skipping Devign row in Table 3")

    return results


def _run_q5(cfg, device):
    import torch

    from data.graph_builder import EDGE_TYPES
    from data.word2vec_embed import NodeFeaturizer
    from evaluation.cve_eval import evaluate_holdout
    from models.devign import build_model

    model_path = os.path.join(artifact_dir(cfg, "devign", "combined"), "model.pt")
    if not os.path.exists(model_path):
        print("[reproduce] no Combined Devign model found; skipping Q5")
        return None
    proc = cfg["data"]["processed_dir"]
    featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
    model = build_model("devign", cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=len(featurizer.type_vocab), num_edge_types=len(EDGE_TYPES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    from scripts.train import load_threshold
    return evaluate_holdout(cfg, model, device,
                            threshold=load_threshold(cfg, "devign", "combined"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="few epochs for a smoke run")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training epochs for every model in this run "
                         "(config.yaml keeps the paper's 200/patience-100 schedule untouched)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-leakage-check", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo every stage, ignoring cached artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])
    epochs = args.epochs if args.epochs is not None else (8 if args.quick else None)
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
        cfg["training"]["early_stopping_patience"] = min(
            cfg["training"]["early_stopping_patience"], max(5, epochs // 4))
    artifacts = ensure_dir(cfg["project"]["artifacts_dir"])
    use_test_split = not cfg["data"].get("paper_split", False)

    print("=" * 70)
    print("STEP 1/6: data preparation")
    train_path = os.path.join(cfg["data"]["processed_dir"], "train.pkl")
    if _stage_done(train_path, args.force):
        print("[reproduce] skipped (data/processed already built; pass --force to rebuild)")
    else:
        info = prepare(cfg, verbose=True)
        print("[reproduce] prepared:", info)

    print("=" * 70)
    print("STEP 2/6: train Devign + Ggrn (per-project + Combined)")
    graph_results = _run_graph_matrix(cfg, device, epochs, args.force)

    baseline_results, baseline_failures = {}, []
    if not args.skip_baselines:
        print("=" * 70)
        print("STEP 3/6: train baselines (per-project + Combined)")
        baseline_results, baseline_failures = _run_baseline_matrix(cfg, device, args.force)
        if baseline_failures:
            print(f"[reproduce] {len(baseline_failures)} baseline run(s) failed: "
                  f"{baseline_failures}")

    table2 = _assemble_table2(cfg, graph_results, baseline_results, use_test_split)
    print("\n" + format_table2(table2, cfg["data"]["projects"]))
    with open(os.path.join(artifacts, "table2.json"), "w") as f:
        json.dump(table2, f, indent=2)

    print("=" * 70)
    print("STEP 4/6: imbalanced (Table 3) + unseen-commit holdout (Q5)")
    table3 = _run_imbalanced(cfg, device)
    print(format_table3(table3, cfg["data"]["projects"]))
    with open(os.path.join(artifacts, "table3.json"), "w") as f:
        json.dump(table3, f, indent=2)

    q5 = _run_q5(cfg, device)
    if q5:
        with open(os.path.join(artifacts, "cve.json"), "w") as f:
            json.dump(q5, f, indent=2)
        if q5.get("num_functions"):
            print(f"[reproduce] Q5 unseen-commit holdout: {q5['overall_accuracy']:.2f}% "
                  f"over {q5['num_functions']} functions")

    if not args.skip_ablation:
        print("=" * 70)
        print("STEP 5/6: single-edge ablation (Q3, Combined)")
        abl_path = os.path.join(artifacts, ABLATION_ARTIFACT_PATH)
        if _stage_done(abl_path, args.force):
            print("[reproduce] skipped (ablation.json already present)")
        else:
            abl = run_ablation(cfg, ["devign", "ggrn"], device, epochs=epochs, project="combined")
            from evaluation.report import format_ablation
            print(format_ablation(abl))
            ensure_dir(os.path.dirname(abl_path))
            with open(abl_path, "w") as f:
                json.dump(abl, f, indent=2)

    if not args.skip_leakage_check:
        print("=" * 70)
        print("STEP 6/6: commit-disjoint leakage check")
        leakage_path = os.path.join(artifacts, "leakage_check.json")
        if _stage_done(leakage_path, args.force):
            print("[reproduce] skipped (leakage_check.json already present)")
        else:
            from scripts.run_leakage_check import _leakage_config
            from scripts.run_leakage_check import main as leakage_main
            import sys
            argv = sys.argv
            sys.argv = ["run_leakage_check", "--config", args.config] + \
                       (["--epochs", str(epochs)] if epochs else [])
            try:
                leakage_main()
            finally:
                sys.argv = argv

    print("=" * 70)
    print("Reproduction complete. Artifacts in", artifacts)
    if baseline_failures:
        raise SystemExit(f"{len(baseline_failures)} baseline run(s) failed: {baseline_failures}")


if __name__ == "__main__":
    main()
