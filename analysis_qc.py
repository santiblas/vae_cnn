#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_qc.py

Módulo de Control de Calidad (QC) Post-Hoc para el pipeline VAE-Clasificador.

Funciones principales:
- summarize_distribution_stages:
    Compara las distribuciones de datos (crudo, normalizado, reconstruido)
    para validar la coherencia del preprocesamiento y la arquitectura del VAE.
- evaluate_scanner_leakage:
    Evalúa el "batch effect" (fuga de información del escáner/sitio de adquisición)
    en espacio conectómico normalizado y en el espacio latente.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _offdiag_mask(n: int) -> np.ndarray:
    """
    Máscara booleana off-diagonal (True fuera de la diagonal).
    """
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m


def _sanitize_name(name: str) -> str:
    """
    Limpia nombres de canal para usarlos en nombres de archivo.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", name)[:80]


def _compute_offdiag_values(
    tensor_4d: np.ndarray,
    channel_idx: int,
    offdiag_mask: np.ndarray
) -> np.ndarray:
    """
    Extrae TODOS los valores off-diagonal para un canal específico,
    apilando todos los sujetos.

    tensor_4d shape esperada: [N_subjects, N_channels, R, R]
    """
    vals = tensor_4d[:, channel_idx, :, :][:, offdiag_mask].ravel()
    return vals


def _stage_stats_df(
    tensor_4d: np.ndarray,
    channel_names: List[str],
    final_activation: Optional[str] = None
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por canal (solo off-diagonal):
    - moments básicos (mean, std, skew, kurtosis)
    - percentiles
    - colas extremas
    - saturación si aplica (tanh / sigmoid / linear)
    """
    n_subj, n_ch, n_rois, _ = tensor_4d.shape
    offmask = _offdiag_mask(n_rois)
    stats_list: List[Dict] = []

    for c in range(n_ch):
        vals = _compute_offdiag_values(tensor_4d, c, offmask)
        if vals.size == 0:
            continue

        prc = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])

        row = {
            "channel": channel_names[c] if c < len(channel_names) else f"Chan{c}",
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skew": float(skew(vals, bias=False)),
            "kurtosis": float(kurtosis(vals, fisher=False, bias=False)),
            "min": float(np.min(vals)),
            "p1": float(prc[0]),
            "p5": float(prc[1]),
            "p25": float(prc[2]),
            "p50": float(prc[3]),
            "p75": float(prc[4]),
            "p95": float(prc[5]),
            "p99": float(prc[6]),
            "max": float(np.max(vals)),
            "frac_abs_gt2": float(np.mean(np.abs(vals) > 2.0)),
            "frac_abs_gt3": float(np.mean(np.abs(vals) > 3.0)),
        }

        # saturación / clipping según activación final del decoder
        if final_activation is not None:
            if final_activation == "tanh":
                row["frac_saturation"] = float(np.mean(np.abs(vals) > 0.95))
            elif final_activation == "sigmoid":
                row["frac_saturation"] = float(np.mean((vals < 0.05) | (vals > 0.95)))
            elif final_activation == "linear":
                # sin saturación teórica; reportamos valores hiper-extremos
                row["frac_saturation"] = float(np.mean(np.abs(vals) > 5.0))

        stats_list.append(row)

    return pd.DataFrame(stats_list)


def _save_overlay_hist_per_channel(
    raw_tensor: np.ndarray,
    norm_tensor: np.ndarray,
    recon_tensor: np.ndarray,
    channel_names: List[str],
    out_dir: Path,
    prefix: str
) -> None:
    """
    Para cada canal: histograma raw vs norm vs recon.
    Guarda .png por canal en out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _, n_ch, n_rois, _ = raw_tensor.shape
    offmask = _offdiag_mask(n_rois)

    for c in range(n_ch):
        ch_name = channel_names[c] if c < len(channel_names) else f"Chan{c}"

        raw_vals = _compute_offdiag_values(raw_tensor, c, offmask)
        norm_vals = _compute_offdiag_values(norm_tensor, c, offmask)
        recon_vals = _compute_offdiag_values(recon_tensor, c, offmask)

        # downsample para que matplotlib no muera con decenas de millones de puntos
        if raw_vals.size > 2_000_000:
            raw_vals = np.random.choice(raw_vals, 2_000_000, replace=False)
        if norm_vals.size > 2_000_000:
            norm_vals = np.random.choice(norm_vals, 2_000_000, replace=False)
        if recon_vals.size > 2_000_000:
            recon_vals = np.random.choice(recon_vals, 2_000_000, replace=False)

        plt.figure(figsize=(10, 6))
        plt.hist(
            raw_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Raw (Std={np.std(raw_vals):.2f})",
            color="gray"
        )
        plt.hist(
            norm_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Norm (Std={np.std(norm_vals):.2f})",
            color="blue"
        )
        plt.hist(
            recon_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Recon (Std={np.std(recon_vals):.2f})",
            color="red"
        )
        plt.xlabel("Valor de Conectividad (off-diag)")
        plt.ylabel("Densidad")
        plt.title(f"Superposición de Distribuciones: {ch_name} ({prefix})")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        fname = out_dir / f"{prefix}_hist_{_sanitize_name(ch_name)}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def summarize_distribution_stages(
    raw_tensor: np.ndarray,
    norm_tensor: np.ndarray,
    recon_tensor: np.ndarray,
    channel_names: List[str],
    out_dir: Path,
    prefix: str,
    final_activation: str
) -> Dict[str, pd.DataFrame]:
    """
    Calcula stats canal-wise para raw/norm/recon y guarda:
    - CSVs con stats
    - PNGs con histogramas overlaid
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = _stage_stats_df(
        raw_tensor,
        channel_names,
        final_activation=None
    )
    df_norm = _stage_stats_df(
        norm_tensor,
        channel_names,
        final_activation=None
    )
    df_rec = _stage_stats_df(
        recon_tensor,
        channel_names,
        final_activation=final_activation
    )

    df_raw.to_csv(out_dir / f"{prefix}_dist_raw.csv", index=False)
    df_norm.to_csv(out_dir / f"{prefix}_dist_norm.csv", index=False)
    df_rec.to_csv(out_dir / f"{prefix}_dist_recon.csv", index=False)

    _save_overlay_hist_per_channel(
        raw_tensor,
        norm_tensor,
        recon_tensor,
        channel_names,
        out_dir,
        prefix,
    )

    return {"raw": df_raw, "norm": df_norm, "recon": df_rec}


def evaluate_scanner_leakage(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    normalized_tensor_subjects: np.ndarray,
    latent_mu_subjects: np.ndarray,
    out_dir: Path,
    fold_tag: str,
    random_state: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Evalúa qué tan separable es el sitio/escáner (batch effect) con:
      A) conectomas normalizados aplanados
      B) espacio latente mu del VAE

    Usa LogisticRegression con CV (balanced_accuracy). Guarda CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # buscar columna de sitio/escáner
    site_cols = [
        c for c in metadata_df_full.columns
        if ("Manufacturer" in c.lower()) or ("vendor" in c.lower()) or ("site" in c.lower())
    ]
    if len(site_cols) == 0:
        print(f"[{fold_tag} QC] No se encontró columna de escáner/sitio. Saltando análisis de fuga.")
        return None
    site_col = site_cols[0]

    # mapear tensor_idx -> etiqueta de sitio
    idx_to_site = dict(
        zip(
            metadata_df_full["tensor_idx"].values,
            metadata_df_full[site_col].astype(str).values
        )
    )
    y_site = np.array([idx_to_site[i] for i in subject_global_indices])

    if len(np.unique(y_site)) < 2:
        print(f"[{fold_tag} QC] Solo se encontró 1 sitio/escáner. No se puede estimar fuga.")
        return None

    # features conectoma normalizado (flatten)
    N, C, R, _ = normalized_tensor_subjects.shape
    X_conn = normalized_tensor_subjects.reshape(N, C * R * R)

    # features latentes
    X_lat = latent_mu_subjects

    clf_site = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        class_weight="balanced",
        solver="lbfgs"
    )
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    acc_conn = cross_val_score(
        clf_site, X_conn, y_site, cv=cv_inner, scoring="balanced_accuracy"
    )
    acc_lat = cross_val_score(
        clf_site, X_lat, y_site, cv=cv_inner, scoring="balanced_accuracy"
    )

    df_out = pd.DataFrame({
        "representation": ["connectome_norm", "latent_mu"],
        "balanced_accuracy_mean": [float(np.mean(acc_conn)), float(np.mean(acc_lat))],
        "balanced_accuracy_std":  [float(np.std(acc_conn)),  float(np.std(acc_lat))],
        "n_classes": [int(len(np.unique(y_site)))] * 2,
        "site_col": [site_col] * 2,
        "fold_tag": [fold_tag] * 2,
    })

    df_out.to_csv(out_dir / f"{fold_tag}_scanner_leakage.csv", index=False)
    return df_out
