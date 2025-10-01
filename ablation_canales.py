#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_canales.py
-------------------

Orquestador para análisis de ablación de canales sobre serentipia9.py.

Estrategia:
1) Evalúa TODOS los canales individualmente (1 a la vez).
2) Arranca con el mejor canal (según métrica elegida).
3) En cada paso agrega greedy el canal que más mejora la métrica.
4) Se detiene si no hay mejora mayor a un umbral (o sigue hasta usar todos).

Requisitos:
- Tu serentipia9.py accesible (mismo entorno con PyTorch, Optuna, etc.).
- Rutas válidas para global tensor, metadata y roi_order.

Autor: tú + yo :)
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import time
import json
import csv
import re
import numpy as np
import pandas as pd

# Mapea índice->nombre solo para imprimir bonito (ajusta si tu orden difiere)
DEFAULT_CHANNEL_NAMES = [
    'Pearson_OMST_GCE_Signed_Weighted',
    'Pearson_Full_FisherZ_Signed',
    'MI_KNN_Symmetric',
    'dFC_AbsDiffMean',
    'dFC_StdDev',
    'DistanceCorr',
    'Granger_F_lag1',
]

# ---------- utilidades de sistema ----------

def run_cmd(cmd: list[str], cwd: str | None = None) -> int:
    print("\n[RUN]", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode

def newest_metrics_csv(output_dir: Path) -> Path | None:
    # Busca el CSV de métricas que serentipia9.py genera (all_folds_metrics_MULTI_*.csv)
    cands = sorted(output_dir.glob("all_folds_metrics_MULTI_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def read_metric_from_csv(csv_path: Path, classifier: str, metric: str) -> tuple[float, float]:
    """
    Devuelve (mean, std) para `metric` filtrando por 'actual_classifier_type'==classifier.
    Si hay varias filas (folds), promedia.
    """
    df = pd.read_csv(csv_path)
    if "actual_classifier_type" in df.columns:
        df = df[df["actual_classifier_type"] == classifier]
    if df.empty or metric not in df.columns:
        return (float("nan"), float("nan"))
    return (float(df[metric].mean()), float(df[metric].std()))

def pretty_channels(indices: list[int]) -> str:
    names = []
    for i in indices:
        try:
            names.append(f"{i}:{DEFAULT_CHANNEL_NAMES[i]}")
        except Exception:
            names.append(f"{i}:Channel{i}")
    return "[" + ", ".join(names) + "]"

# ---------- runner de una corrida ----------

def run_serentipia_once(
    channels: list[int],
    args_base: dict,
    run_root: Path,
    run_tag: str,
    classifier: str,
) -> dict:
    """
    Lanza una corrida de serentipia9.py con un set de canales.
    Guarda resultados en run_root/run_tag/.
    Retorna dict con info de corrida (path, métricas, etc.).
    """
    out_dir = run_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construye la línea de comando (basada en tu ejemplo; podés ajustar defaults aquí)
    cmd = [
        sys.executable, args_base["serentipia_path"],
        "--global_tensor_path", args_base["global_tensor_path"],
        "--metadata_path", args_base["metadata_path"],
        "--roi_order_path", args_base["roi_order_path"],
        "--output_dir", str(out_dir),

        # FIX: siempre usar Age y Sex como metadatos
        "--metadata_features", "Age", "Sex",

        # Clasificador: uno solo para estabilidad de la ablación
        "--classifier_types", classifier,
        "--gridsearch_scoring", args_base.get("gridsearch_scoring", "roc_auc"),
    ]

    # Hiperparams razonables por defecto (copiados de tu corrida de ejemplo)
    # Preset rápido pensado para ranking de canales (sin buscar SOTA)
    if args_base.get("fast", False):
        # 👉 En modo rápido, extendemos el comando directamente
        cmd += [
            "--outer_folds", str(args_base.get("outer_folds", 3)),
            "--repeated_outer_folds_n_repeats", "1",
            "--epochs_vae", str(args_base.get("epochs_vae_fast", 250)),
            "--early_stopping_patience_vae", "10",
            "--cyclical_beta_n_cycles", "2",
            "--lr_scheduler_type", "cosine_warm",
            "--lr_scheduler_T0", "10",
            "--lr_scheduler_eta_min", "1e-6",
            "--batch_size", "128",
            "--beta_vae", "1.2",
            "--dropout_rate_vae", str(args_base.get("dropout_vae", 0.2)),
            "--latent_dim", str(args_base.get("latent_dim_fast", 512)),
            "--n_jobs_gridsearch", "1",
            "--optuna_max_trials", str(args_base.get("optuna_max_trials", 10)),
            "--optuna_timeout", str(args_base.get("optuna_timeout", 120)),
            "--classifier_use_class_weight",
            "--clf_no_tune",
            "--no_smote",
            "--fast_ablation",
        ]
    else:
        extra_flags = [
            ("--outer_folds", "outer_folds", 5),
            ("--repeated_outer_folds_n_repeats", "repeats", 1),
            ("--epochs_vae", "epochs_vae", 300),
            ("--early_stopping_patience_vae", "early_stop", 30),
            ("--cyclical_beta_n_cycles", "beta_cycles", 4),
            ("--lr_scheduler_type", "lr_sched_type", "cosine_warm"),
            ("--lr_scheduler_T0", "lr_sched_T0", 30),
            ("--lr_scheduler_eta_min", "lr_sched_eta_min", 5e-7),
            ("--batch_size", "batch_size", 64),
            ("--beta_vae", "beta_vae", 4.6),
            ("--dropout_rate_vae", "dropout_vae", 0.2),
            ("--latent_dim", "latent_dim", 512),
            ("--n_jobs_gridsearch", "n_jobs_gridsearch", 8),
            ("--optuna_max_trials", "optuna_max_trials", 80),
            ("--optuna_timeout", "optuna_timeout", 1800),
        ]
        for flag, key, default in extra_flags:
            cmd.extend([flag, str(args_base.get(key, default))])

    # Canales
    if not channels:
        raise ValueError("Lista de canales vacía.")
    cmd += ["--channels_to_use"] + [str(i) for i in channels]

    # Flags booleanos (se añaden solo si están en args_base y son True)
    bool_flags = {
        "--use_smote": "use_smote",
        "--use_optuna_pruner": "use_optuna_pruner",
        "--classifier_use_class_weight": "classifier_use_class_weight",
        "--save_fold_artefacts": "save_fold_artefacts",
        "--save_vae_training_history": "save_vae_training_history",
        "--clf_no_tune": "clf_no_tune",
        "--no_smote": "no_smote",
    }
    for flag, key in bool_flags.items():
        if args_base.get(key, False):
            cmd.append(flag)

    # Ejecuta si no existe ya el CSV (permite reusar resultados)
    metrics_csv = newest_metrics_csv(out_dir)
    if metrics_csv is None:
        rc = run_cmd(cmd)
        if rc != 0:
            return {"ok": False, "error": f"serentipia9.py retornó código {rc}", "out_dir": str(out_dir)}
        metrics_csv = newest_metrics_csv(out_dir)

    if metrics_csv is None:
        return {"ok": False, "error": "No se encontró metrics CSV tras la ejecución.", "out_dir": str(out_dir)}

    # Lee métrica elegida
    metric = args_base.get("opt_metric", "auc")
    mean_val, std_val = read_metric_from_csv(metrics_csv, classifier=classifier, metric=metric)

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "metrics_csv": str(metrics_csv),
        "metric_name": metric,
        "metric_mean": mean_val,
        "metric_std": std_val,
        "channels": list(channels),
        "classifier": classifier,
    }

# ---------- algoritmo greedy de ablación ----------

def greedy_ablation(
    candidate_channels: list[int],
    args_base: dict,
    run_root: Path,
    classifier: str,
    min_improvement: float,
    stop_when_no_improve: bool,
) -> list[dict]:
    """
    1) Evalúa todos los canales single y elige el mejor como semilla.
    2) Iterativamente prueba agregar cada canal restante; elige el que más suba la métrica.
    3) Se detiene si no mejora >= min_improvement (si stop_when_no_improve).
    4) Devuelve lista de pasos con resultados.
    """
    results: list[dict] = []

    # Paso 0: singles
    singles = []
    for ch in candidate_channels:
        tag = f"single_ch{ch}"
        res = run_serentipia_once([ch], args_base, run_root, tag, classifier)
        if not res.get("ok", False):
            print(f"[WARN] single canal {ch} falló: {res.get('error')}")
            continue
        print(f"[SINGLE] ch {pretty_channels([ch])} -> {res['metric_name']}={res['metric_mean']:.4f} ±{res['metric_std']:.4f}")
        singles.append(res)

    if not singles:
        print("[ERROR] No hay resultados de singles. Abortando.")
        return results

    # Mejor single
    best_single = max(singles, key=lambda r: (r["metric_mean"], -len(r["channels"])))
    picked = list(best_single["channels"])
    results.append(best_single)
    remaining = [c for c in candidate_channels if c not in picked]

    print(f"\n[SEED] Mejor canal inicial: {pretty_channels(picked)} "
          f"con {best_single['metric_name']}={best_single['metric_mean']:.4f}")

    # Pasos greedy
    step = 1
    while remaining:
        print(f"\n[STEP {step}] Conjunto actual: {pretty_channels(picked)}")
        trials = []
        for cand in remaining:
            trial_channels = picked + [cand]
            tag = f"step{step}_add_ch{cand}_set_" + "_".join(map(str, trial_channels))
            res = run_serentipia_once(trial_channels, args_base, run_root, tag, classifier)
            if res.get("ok", False):
                print(f"  + probar {pretty_channels([cand])} -> {res['metric_name']}={res['metric_mean']:.4f}")
                trials.append(res)
            else:
                print(f"  ! fallo con canal {cand}: {res.get('error')}")

        if not trials:
            print("[INFO] No hubo trials válidos. Corto aquí.")
            break

        best_trial = max(trials, key=lambda r: r["metric_mean"])
        gain = best_trial["metric_mean"] - results[-1]["metric_mean"]
        print(f"[CHOICE] agrego canal {pretty_channels([c for c in best_trial['channels'] if c not in picked])} "
              f"→ {best_trial['metric_name']}={best_trial['metric_mean']:.4f} (Δ={gain:+.4f})")

        results.append(best_trial)
        picked = list(best_trial["channels"])
        remaining = [c for c in candidate_channels if c not in picked]

        if stop_when_no_improve and gain < min_improvement:
            print(f"[STOP] Mejora {gain:.4f} < umbral {min_improvement:.4f}. Detengo.")
            break

        step += 1

    return results

# ---------- main / CLI ----------

def main():
    p = argparse.ArgumentParser(description="Ablación de canales (greedy) encima de serentipia9.py")
    p.add_argument("--fast", action="store_true",
                   help="Preset rápido: fuerza setup veloz para ranking de canales (logreg sin tuning, VAE corto).")
    p.add_argument("--serentipia_path", type=str, default="serentipia9.py", help="Ruta a serentipia9.py")
    p.add_argument("--global_tensor_path", type=str, required=True)
    p.add_argument("--metadata_path", type=str, required=True)
    p.add_argument("--roi_order_path", type=str, required=True)

    p.add_argument("--output_root", type=str, default="./ablation_runs", help="Carpeta raíz de resultados")
    p.add_argument("--classifier", type=str, default="mlp",
                   choices=["rf","gb","svm","logreg","mlp","xgb","cat"],
                   help="Clasificador a usar en TODAS las corridas")
    p.add_argument("--metric", type=str, default="auc", help="Columna de métrica a optimizar del CSV (p.ej. auc, pr_auc, balanced_accuracy)")
    p.add_argument("--gridsearch_scoring", type=str, default="roc_auc", help="Scoring para OptunaSearchCV")

    p.add_argument("--candidate_channels", type=int, nargs="*", default=None,
                   help="Índices de canales a considerar. Por defecto usa todos los detectados en el tensor.")
    p.add_argument("--min_improvement", type=float, default=0.001, help="Umbral de mejora mínima (AUC) para continuar")
    p.add_argument("--no_early_stop", action="store_true", help="No cortar si no mejora; continúa hasta usar todos")

    # Presets de entrenamiento (podés cambiarlos si querés acelerar)
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--epochs_vae", type=int, default=300)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--beta_cycles", type=int, default=4)
    p.add_argument("--lr_sched_type", type=str, default="cosine_warm", choices=["plateau","cosine_warm"])
    p.add_argument("--lr_sched_T0", type=int, default=30)
    p.add_argument("--lr_sched_eta_min", type=float, default=5e-7)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--beta_vae", type=float, default=4.6)
    p.add_argument("--dropout_vae", type=float, default=0.2)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--n_jobs_gridsearch", type=int, default=8)

    args = p.parse_args()

    run_root = Path(args.output_root)
    run_root.mkdir(parents=True, exist_ok=True)


    # Descubrir número de canales si no se provee lista
    if args.candidate_channels is None:
        # Leemos la forma del npz para saber cuántos canales hay
        npz = np.load(args.global_tensor_path)
        tensor = npz['global_tensor_data']
        _, C, _, _ = tensor.shape
        candidate_channels = list(range(C))
    else:
        candidate_channels = list(args.candidate_channels)

    args_base = vars(args).copy()
    if args.fast:
        print("[INFO] Modo --fast activado. Usando configuración rápida.")
        args_base.update({
            "classifier": "logreg",
            "metric": "auc",
            "gridsearch_scoring": "roc_auc",
            "clf_no_tune": True,
            "no_smote": True,
            "epochs_vae": 200,
            "early_stop": 10,
            "beta_cycles": 2,
            "lr_sched_T0": 10,
            "latent_dim_fast": 128,   # usa *_fast porque tu runner los lee así
            "batch_size": 128,
            "fast": True,
            "optuna_max_trials": 10,
            "optuna_timeout": 120,
            "n_jobs_gridsearch": 1,
        })

    # en vez de reemplazar, añade/actualiza campos faltantes
    args_base.update({
        "serentipia_path": args.serentipia_path,
        "global_tensor_path": args.global_tensor_path,
        "metadata_path": args.metadata_path,
        "roi_order_path": args.roi_order_path,
        "outer_folds": args.outer_folds,
        "repeats": args.repeats,
        "epochs_vae": args.epochs_vae,
        "early_stop": args.early_stop,
        "beta_cycles": args.beta_cycles,
        "lr_sched_type": args.lr_sched_type,
        "lr_sched_T0": args.lr_sched_T0,
        "lr_sched_eta_min": args.lr_sched_eta_min,
        "batch_size": args.batch_size,
        "beta_vae": args.beta_vae,
        "dropout_vae": args.dropout_vae,
        "latent_dim": args.latent_dim,
        "n_jobs_gridsearch": args.n_jobs_gridsearch,
        "gridsearch_scoring": args.gridsearch_scoring,
        "opt_metric": args.metric,
    })
    # Asegurá que pasen estos booleanos a run_serentipia_once
    args_base.setdefault("use_smote", False if args.fast else True)
    args_base.setdefault("no_smote", True if args.fast else False)
    #args.classifier = "logreg" # Actualizar para el bucle principal

    print("\n===== ABLAción de canales =====")
    print(f"Clasificador fijo: {args.classifier}")
    print(f"Canales candidatos: {pretty_channels(candidate_channels)}")
    print(f"Salida raíz: {run_root.resolve()}\n")

    # En modo rápido, si te olvidaste de cambiar, usa logreg por defecto
    if args.fast and args.classifier == "mlp":
        args.classifier = "logreg"

    results = greedy_ablation(
        candidate_channels=candidate_channels,
        args_base=args_base,
        run_root=run_root,
        classifier=args.classifier,
        min_improvement=args.min_improvement,
        stop_when_no_improve=not args.no_early_stop,
    )

    if not results:
        print("\n[FIN] No hubo resultados para resumir.")
        return

    # Guardar resumen global
    summary_path = run_root / "summary_ablation.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step",
            "channels_indices",
            "channels_pretty",
            "classifier",
            "metric",
            "metric_mean",
            "metric_std",
            "delta_vs_prev",
            "out_dir",
            "metrics_csv",
        ])
        prev = None
        for i, res in enumerate(results, start=0):
            delta = "" if prev is None else f"{(res['metric_mean'] - prev):+.6f}"
            w.writerow([
                i,
                " ".join(map(str, res["channels"])),
                pretty_channels(res["channels"]),
                res["classifier"],
                res["metric_name"],
                f"{res['metric_mean']:.6f}",
                f"{res['metric_std']:.6f}",
                delta,
                res["out_dir"],
                res["metrics_csv"],
            ])
            prev = res["metric_mean"]

    print(f"\n[OK] Resumen guardado en: {summary_path.resolve()}")
    print("Contenido (corto):")
    df = pd.read_csv(summary_path)
    with pd.option_context("display.max_colwidth", 120):
        print(df[["step","channels_pretty","metric","metric_mean","metric_std","delta_vs_prev"]])

if __name__ == "__main__":
    main()
