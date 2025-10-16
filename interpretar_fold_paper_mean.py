#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpretar_fold_paper_mean.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from typing import Optional
import os
import random

try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("WARN: 'captum' no encontrado. 'integrated_gradients' no disponible. Instala: pip install captum")
    IntegratedGradients = None

# ---------------------------------------------------------------------------
# Imports de tu repo local
# ---------------------------------------------------------------------------
try:
    from models.convolutional_vae2 import ConvolutionalVAE
except ImportError as e:  # ayuda si se ejecuta fuera del repo raíz
    raise ImportError("No se pudo importar ConvolutionalVAE desde models.convolutional_vae2.\n"
                      "Asegúrate de ejecutar desde la raíz del proyecto o de que PYTHONPATH esté configurado.") from e

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('interpret')
logging.getLogger('shap').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception: pass
    random.seed(seed); np.random.seed(seed)
 
# ---------------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------------
import re

def _is_generic_x_names(names: Sequence[str]) -> bool:
    """True si los nombres son del tipo x0, x1, ... (sin semántica)."""
    if len(names) == 0:
        return False
    pat = re.compile(r'^x\d+$')
    return all(isinstance(n, str) and pat.match(n) for n in names)

def _safe_feature_names_after_preproc(
    preproc: Any,
    raw_feature_names: Sequence[str],
    selector: Optional[Any] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Devuelve (feat_names_after, support_mask_optional) de forma robusta:
    - Intenta usar get_feature_names_out con raw_feature_names.
    - Si el resultado parece genérico (x0, x1, ...), hace fallback a raw_feature_names.
    - Aplica el selector (si existe) tanto a los datos como a los nombres.
    """
    raw_feature_names = np.array(list(map(str, raw_feature_names)))
    feat_names_after = None
    try:
        # Muchos transformadores aceptan input_features opcionalmente
        if hasattr(preproc, "get_feature_names_out"):
            try:
                feat_names_after = preproc.get_feature_names_out(raw_feature_names)
            except TypeError:
                feat_names_after = preproc.get_feature_names_out()
    except Exception:
        feat_names_after = None

    if feat_names_after is None or _is_generic_x_names(feat_names_after):
        # Fallback seguro: preservamos el orden original
        feat_names_after = raw_feature_names

    feat_names_after = np.array(list(map(str, feat_names_after)))

    support = None
    if selector is not None and hasattr(selector, "get_support"):
        support = selector.get_support()
        if support is not None and support.dtype == bool and support.shape[0] == feat_names_after.shape[0]:
            feat_names_after = feat_names_after[support]
        else:
            # Si no coincide, mejor no aplicar para no desalinear
            support = None
    return feat_names_after, support

def _build_background_from_train(
    *, fold_dir: Path, cnad_df: pd.DataFrame, tensor_all: np.ndarray, channels: Sequence[int],
    norm_params: List[Dict[str, float]], meta_cols: List[str], vae: ConvolutionalVAE,
    device: torch.device, preproc: Any, selector: Optional[Any], feat_names_target: Sequence[str],
    sample_size: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Genera un background de SHAP crudo (antes de procesar) desde los sujetos de entrenamiento del fold."""
    # 1) Obtener los índices de entrenamiento del clasificador
    test_idx_path = fold_dir / 'test_indices.npy'
    if not test_idx_path.exists():
        raise FileNotFoundError(f"No se encontró {test_idx_path} para reconstruir el background.")
    test_idx_in_cnad = np.load(test_idx_path)

    # El conjunto de entrenamiento son todos los sujetos CN/AD que no están en test
    all_cnad_indices = np.arange(len(cnad_df))
    train_idx_in_cnad = np.setdiff1d(all_cnad_indices, test_idx_in_cnad, assume_unique=True)

    if train_idx_in_cnad.size == 0:
        raise RuntimeError("No hay sujetos de entrenamiento para construir el background.")

    # 2) Tomar una muestra representativa (100-150 es un buen número para SHAP)
    rng = np.random.RandomState(seed)
    if train_idx_in_cnad.size > sample_size:
        train_idx_in_cnad_sample = np.sort(rng.choice(train_idx_in_cnad, size=sample_size, replace=False))
    else:
        train_idx_in_cnad_sample = train_idx_in_cnad
    
    train_df_sample = cnad_df.iloc[train_idx_in_cnad_sample]
    gidx_train_sample = train_df_sample['tensor_idx'].values

    # 3) Reconstruir las features crudas para esta muestra (Tensor -> Latentes -> +Metadatos)
    # Cargar y normalizar tensores
    tens_train_sample = tensor_all[gidx_train_sample][:, channels, :, :]
    tens_train_sample = apply_normalization_params(tens_train_sample, norm_params)
    tens_train_sample_t = torch.from_numpy(tens_train_sample).float().to(device)

    # Obtener latentes del VAE
    with torch.no_grad():
        _, mu, _, z = vae(tens_train_sample_t)
    lat_np = mu.detach().cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat_train_sample = pd.DataFrame(lat_np, columns=lat_cols)
    
    # Replicar el mapeo de 'Sex' a numérico
    if 'Sex' in meta_cols:
        train_df_sample = train_df_sample.copy()
        train_df_sample.loc[:, 'Sex'] = train_df_sample['Sex'].map({'M': 0, 'F': 1, 'f': 1, 'm': 0})

    # Combinar latentes y metadatos
    X_bg_raw = pd.concat([X_lat_train_sample.reset_index(drop=True),
                          train_df_sample[meta_cols].reset_index(drop=True)],
                         axis=1)

    return X_bg_raw

def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Elimina el prefijo "_orig_mod." que añade torch.compile."""
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def build_vae(vae_kwargs: Dict[str, Any], state_dict_path: Path, device: torch.device) -> ConvolutionalVAE:
    """Construye e inicializa un VAE con pesos entrenados."""
    vae = ConvolutionalVAE(**vae_kwargs).to(device)
    sd = torch.load(state_dict_path, map_location=device)
    vae.load_state_dict(clean_state_dict(sd))
    vae.eval()
    return vae

def _project_to_H(sal_map: np.ndarray) -> np.ndarray:
    """Proyecta un mapa de saliencia al subespacio de matrices huecas y simétricas."""
    # sal_map: (C, R, R)
    sal_sym = 0.5 * (sal_map + sal_map.transpose(0, 2, 1))
    for c in range(sal_sym.shape[0]):
        np.fill_diagonal(sal_sym[c], 0.0)
    return sal_sym.astype(np.float32)


def unwrap_model_for_shap(model: Any, clf_type: str) -> Any:
    """Extrae el estimador base de un CalibratedClassifierCV cuando aplica."""
    if hasattr(model, 'calibrated_classifiers_') and clf_type in {'xgb', 'gb', 'rf', 'lgbm'}:
        cc = model.calibrated_classifiers_[0]
        if hasattr(cc, 'estimator') and cc.estimator is not None:
            return cc.estimator
        if hasattr(cc, 'base_estimator') and cc.base_estimator is not None:
            return cc.base_estimator
    return model

def _grad_to_signed_and_abs(grad_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recibe gradientes por-batch sobre la entrada (B,C,R,R) y devuelve:
      - signed: media (con signo) sobre el batch, proyectada a simétrica y hueca.
      - abs:    |signed|, también proyectada.
    """
    g = grad_batch.detach().mean(dim=0).cpu().numpy()  # (C,R,R)
    signed = _project_to_H(g)
    absmap = _project_to_H(np.abs(g))
    return signed.astype(np.float32), absmap.astype(np.float32)



def _to_sample_feature(sh_vals: Union[np.ndarray, List[np.ndarray]],
                       positive_idx: int,
                       n_samples: int,
                       n_features: int) -> np.ndarray:
    """Devuelve siempre array 2D (samples, features) para la clase positiva."""
    # TreeExplainer binario → array (n_samples, n_features)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 2 and sh_vals.shape == (n_samples, n_features):
        return sh_vals
    # Tree/Kernal multiclase → list de arrays
    if isinstance(sh_vals, list):
        return sh_vals[positive_idx]
    # Formato 3D → (samples, features, classes)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 3:
        return sh_vals[:, :, positive_idx]
    raise ValueError(f"Formato SHAP inesperado: type={type(sh_vals)} shape={getattr(sh_vals,'shape',None)}")


# ---------------------------------------------------------------------------
# Normalización (copiado de serentipia13.py para independencia)
# ---------------------------------------------------------------------------

def apply_normalization_params(data_tensor_subset: np.ndarray,
                               norm_params_per_channel_list: List[Dict[str, float]]) -> np.ndarray:
    """Aplica parámetros de normalización guardados por canal (off-diag)."""
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)
    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError("# canales en datos != # canales en parámetros de normalización")
    for c_idx, params in enumerate(norm_params_per_channel_list):
        mode = params.get('mode', 'zscore_offdiag')
        if params.get('no_scale', False):
            continue
        current_channel_data = data_tensor_subset[:, c_idx, :, :]
        scaled_channel_data = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                std = params.get('std', 1.0)
                mean = params.get('mean', 0.0)
                if std > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mean) / std
            elif mode == 'minmax_offdiag':
                mn = params.get('min', 0.0)
                mx = params.get('max', 1.0)
                rng = mx - mn
                if rng > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mn) / rng
                else:
                    scaled_channel_data[:, off_diag_mask] = 0.0
        normalized_tensor_subset[:, c_idx, :, :] = scaled_channel_data
    return normalized_tensor_subset


# ---------------------------------------------------------------------------
# Carga de datos / artefactos del fold
# ---------------------------------------------------------------------------

def _load_global_and_merge(global_tensor_path: Path, metadata_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Carga tensor global (.npz) + metadata CSV y los une en un DF con tensor_idx."""
    npz = np.load(global_tensor_path)
    tensor_all = npz['global_tensor_data']
    subj_all = npz['subject_ids'].astype(str)
    meta = pd.read_csv(metadata_path)
    meta['SubjectID'] = meta['SubjectID'].astype(str).str.strip()
    tensor_df = pd.DataFrame({'SubjectID': subj_all, 'tensor_idx': np.arange(len(subj_all))})
    merged = tensor_df.merge(meta, on='SubjectID', how='left')
    return tensor_all, merged


def _subset_cnad(merged_df: pd.DataFrame) -> pd.DataFrame:
    return merged_df[merged_df['ResearchGroup_Mapped'].isin(['CN', 'AD'])].reset_index(drop=True)


def _load_label_info(fold_dir: Path) -> Dict[str, Any]:
    p = fold_dir / 'label_mapping.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # fallback: asumir AD=1 CN=0
    log.warning("label_mapping.json no encontrado; se asume CN=0 / AD=1")
    return {'label_mapping': {'CN': 0, 'AD': 1}, 'positive_label_name': 'AD', 'positive_label_int': 1}


# ---------------------------------------------------------------------------
# SHAP (subcomando "shap")
# ---------------------------------------------------------------------------
def cmd_shap(args: argparse.Namespace) -> None:
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    out_dir = fold_dir / 'interpretability_shap'
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[SHAP] fold={args.fold} clf={args.clf}")

    # 1) Pipeline del clasificador (entrenado)
    pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"No se encontró el pipeline del clasificador: {pipe_path}")
    pipe = joblib.load(pipe_path)

    # 2) Artefactos y datos que usaremos tanto para TEST como para background
    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    label_info = _load_label_info(fold_dir)

    # ROI order no es estrictamente necesario para SHAP, pero lo conservamos por compatibilidad
    roi_order_path_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_path_joblib.exists():
        roi_names = joblib.load(roi_order_path_joblib)
    elif args.roi_order_path is not None:
        roi_names = _load_roi_names(Path(args.roi_order_path))
    else:
        roi_names = None  # OK para SHAP

    # Datos globales + merge con metadata
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)

    # Índices de test en el DataFrame CN/AD (orden como en entrenamiento)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    # Derivar índices de TRAIN para robustecer imputaciones de metadatos (sin fuga)
    all_cnad_idx = np.arange(len(cnad))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    # 3) Tensores test normalizados
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    # 4) VAE y latentes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tensor_all.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)
    with torch.no_grad():
        recon, mu, logvar, z = vae(tens_test_t.to(device))
    lat_np = mu.cpu().numpy() if args.latent_features_type == 'mu' else z.cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat = pd.DataFrame(lat_np, columns=lat_cols)

    # Combinar latentes + metadatos para tener X_raw (features crudas)
    meta_cols = args.metadata_features or []

    # --- Robustez metadatos ---
    test_df = test_df.copy()
    if 'Sex' in meta_cols:
        # map solo si es texto; si ya viene numérica, respetar
        if test_df['Sex'].dtype == object:
            test_df.loc[:, 'Sex'] = test_df['Sex'].map({'M': 0, 'F': 1, 'f': 1, 'm': 0})
        test_df.loc[:, 'Sex'] = pd.to_numeric(test_df['Sex'], errors='coerce')
        # imputación con la moda de TRAIN (0/1). Si no hay, caer al 0.
        sex_train = pd.to_numeric(cnad.iloc[train_idx_in_cnad]['Sex'], errors='coerce')
        sex_mode = sex_train.dropna().mode().iloc[0] if not sex_train.dropna().empty else 0.0
        test_df.loc[:, 'Sex'] = test_df['Sex'].fillna(sex_mode).astype(float)
    if 'Age' in meta_cols:
        test_df.loc[:, 'Age'] = pd.to_numeric(test_df['Age'], errors='coerce')
        age_train = pd.to_numeric(cnad.iloc[train_idx_in_cnad]['Age'], errors='coerce')
        age_median = float(age_train.dropna().median()) if not age_train.dropna().empty else float(test_df['Age'].dropna().median())
        test_df.loc[:, 'Age'] = test_df['Age'].fillna(age_median).astype(float)

    X_raw = pd.concat([X_lat.reset_index(drop=True),
                    test_df[meta_cols].reset_index(drop=True)], axis=1)
    log.info(f"[SHAP] X_raw (test) shape={X_raw.shape} (latentes + {len(meta_cols)} metadatos)")

    # 6) Preprocesamiento del pipeline (ya fitted) + nombres robustos
    preproc = pipe.named_steps['scaler']
    selector = pipe.named_steps.get("feature_select") or pipe.named_steps.get("fs")

    # Guardamos nombres crudos antes de preprocesar
    raw_names = np.array(list(map(str, X_raw.columns)))

    # Transformar datos
    X_proc = preproc.transform(X_raw)
    if selector is not None:
        X_proc = selector.transform(X_proc)

    # Nombres tras preproc/selector (robusto; hace fallback si ve x0,x1,…)
    feat_names, support = _safe_feature_names_after_preproc(preproc, raw_names, selector)
    X_proc_df = pd.DataFrame(X_proc, columns=feat_names)

    # Detectar latentes directamente en los NOMBRES FINALES (tras preproc/selector)
    latent_mask_proc, latent_idx_proc, _ = _find_latent_columns(feat_names)
    # Por si los nombres "finales" no conservan 'latent_*', generamos nombres latentes canónicos
    latent_cols_proc = np.array([f"latent_{i}" for i in latent_idx_proc], dtype=object)

 

    if latent_mask_proc.sum() == 0:
        log.error("[SHAP] No se detectaron columnas latentes en las features procesadas. "
                "Revisa el preprocesamiento/nombres de columnas.")
        # ayuda de depuración: mostrar algunos nombres
        log.error(f"[SHAP] Ejemplo de nombres: {list(X_proc_df.columns[:10])}")
    else:
        log.info(f"[SHAP] Latentes detectadas en procesado: {int(latent_mask_proc.sum())} / {len(feat_names)}")


    # 7) Background: cargar o construir AHORA (ya existen cnad, tensor_all, norm_params, vae, device, preproc, selector, feat_names)
    bg_path_raw = fold_dir / f"shap_background_raw_{args.clf}.joblib"
    bg_path_processed = fold_dir / f"shap_background_data_{args.clf}.joblib"

    if bg_path_raw.exists():
        log.info(f"[SHAP] Cargando background RAW desde: {bg_path_raw.name}")
        background_data = joblib.load(bg_path_raw)
    elif bg_path_processed.exists():
        log.warning(f"[SHAP] No hay RAW; uso el procesado: {bg_path_processed.name}")
        background_data = joblib.load(bg_path_processed)
    else:
        log.warning("[SHAP] No hay background en disco. Construyendo uno desde TRAIN…")
        background_data = _build_background_from_train(
            fold_dir=fold_dir,
            cnad_df=cnad,
            tensor_all=tensor_all,
            channels=args.channels_to_use,
            norm_params=norm_params,
            meta_cols=meta_cols,
            vae=vae,
            device=device,
            preproc=preproc,
            selector=selector,
            feat_names_target=feat_names,
            sample_size=min(100, len(cnad))
        )
        joblib.dump(background_data, bg_path_processed)
        log.info(f"[SHAP] Background procesado guardado en {bg_path_processed.name}")

    # Asegurar que el background tenga mismas columnas/orden que el clasificador
    background_proc = _ensure_background_processed(background_data, preproc, feat_names, selector)

    # 8) Modelo a explicar y SHAP
    model = unwrap_model_for_shap(pipe.named_steps['model'], args.clf)

    # Bloque corregido para interpretar_fold_paper.py

    if args.clf in {'xgb', 'gb', 'rf', 'lgbm'}:
        explainer = shap.TreeExplainer(model, background_proc)
        shap_all = explainer.shap_values(X_proc_df)
    else:
        log.warning("[SHAP] Usando KernelExplainer con masker estable (si está disponible).")
        # Resumimos el background para estabilidad y coste
        k = min(50, len(background_proc))
        np.random.seed(args.seed)
        log.info(f"[SHAP] Resumiendo background de {len(background_proc)} → {k} centroides (kmeans).")
        summary = shap.kmeans(background_proc, k)
        # SHAP < 0.41: kmeans devuelve DenseData; quedarnos con .data (np.ndarray)
        bg_np = getattr(summary, "data", summary)
        # Explainer moderno con masker; si falla (versiones antiguas), fallback a KernelExplainer clásico
        try:
            masker = shap.maskers.Independent(bg_np)
            explainer = shap.Explainer(model.predict_proba, masker, algorithm="kernel")
            exp = explainer(X_proc_df.values, max_evals=args.kernel_nsamples)
            shap_all = exp.values
            base_val = exp.base_values
        except Exception as e:
            log.warning(f"[SHAP] shap.Explainer no disponible/compatible ({e}); fallback a KernelExplainer clásico.")
            explainer = shap.KernelExplainer(model.predict_proba, bg_np)
            shap_all = explainer.shap_values(X_proc_df.values, nsamples=args.kernel_nsamples)
            base_val = explainer.expected_value

    # 9) Clase positiva y empaquetado
    classes_ = list(model.classes_) if hasattr(model, 'classes_') else [0, 1]
    pos_int = label_info['positive_label_int']
    pos_idx = classes_.index(pos_int)
    shap_pos = _to_sample_feature(shap_all, pos_idx, *X_proc_df.shape)

    # base_value ya resuelto arriba si usamos Explainer moderno; si venimos del
    # árbol o KernelExplainer clásico, adaptar al formato (lista por clase)
    if not isinstance(locals().get('base_val', None), (int, float, np.floating)):
        base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[pos_idx]

    pack = {
        'shap_values': shap_pos.astype(np.float32),
        'base_value': float(base_val),
        'X_test': X_proc_df,
        'feature_names': feat_names.tolist(),
        'latent_feature_mask': latent_mask_proc.astype(bool),
        'latent_feature_indices': [int(i) for i in latent_idx_proc],  # índices latentes (enteros)
        'latent_feature_names': latent_cols_proc.tolist(),            # nombres de columnas latentes
        'test_subject_ids': test_df['SubjectID'].astype(str).tolist(),
        'test_labels': test_df['ResearchGroup_Mapped'].map({'CN': 0, 'AD': 1}).astype(int).tolist(),
        'latent_features_type': args.latent_features_type,
        'metadata_features': meta_cols,
        'seed_used': int(args.seed),
    }
    pack_path = out_dir / f'shap_pack_{args.clf}.joblib'
    joblib.dump(pack, pack_path)
    log.info(f"[SHAP] Pack guardado: {pack_path}")

    _plot_shap_summary(shap_pos, X_proc_df, out_dir, args.fold, args.clf, base_val)


def _find_latent_columns(feature_names: Sequence[str]) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Devuelve:
      - mask booleano de columnas latentes,
      - lista con los índices latentes (enteros) detectados,
      - array con los nombres originales de columnas que contienen 'latent_<idx>'
    Acepta nombres tipo: 'latent_12', 'num__latent_12', 'scaler__latent_12', etc.
    """
    import re
    pat = re.compile(r'latent_(\d+)\b')
    mask = np.zeros(len(feature_names), dtype=bool)
    latent_indices: List[int] = []
    cols_matched: List[str] = []
    for i, name in enumerate(map(str, feature_names)):
        m = pat.search(name)
        if m:
            mask[i] = True
            latent_indices.append(int(m.group(1)))
            cols_matched.append(name)
    return mask, latent_indices, np.array(cols_matched, dtype=object)


# ---------------------------------------------------------------------------
# SALIENCIA (subcomando "saliency")
# ---------------------------------------------------------------------------

def get_latent_weights_from_pack(pack: Dict[str, Any], mode: str, top_k: Optional[int]) -> pd.DataFrame:
    """Calcula pesos de las features latentes a partir de un shap_pack.

    mode:
        * mean_abs         → media de |SHAP| en todos los sujetos.
        * mean_signed      → media signed en todos los sujetos.
        * ad_vs_cn_diff    → (media SHAP AD) − (media SHAP CN)  (recomendado).
    """
    shap_values = pack['shap_values']            # (N, F)
    feature_names = pack['feature_names']        # list len=F
    labels = np.asarray(pack['test_labels'])     # (N,)

    # Preferir la máscara latente guardada en el pack (más robusta);
    # si no existe, fallback regex.
    if 'latent_feature_mask' in pack and isinstance(pack['latent_feature_mask'], (list, np.ndarray)):
        latent_mask = np.asarray(pack['latent_feature_mask'], dtype=bool)
        # Alinear por seguridad si la longitud difiere
        if latent_mask.shape[0] != len(feature_names):
            log.warning("[SALIENCY] Máscara latente del pack no alinea; se usará fallback regex.")
            import re
            latent_mask = np.array([bool(re.search(r'(?:^|__)latent_\d+$', n)) for n in feature_names])
    else:
        import re
        latent_mask = np.array([bool(re.search(r'(?:^|__)latent_\d+$', n)) for n in feature_names])

    latent_vals = shap_values[:, latent_mask]
    latent_names = np.array(feature_names)[latent_mask]

    if mode == 'mean_abs':
        importance = np.abs(latent_vals).mean(axis=0)
    elif mode == 'mean_signed':
        importance = latent_vals.mean(axis=0)
    elif mode == 'ad_vs_cn_diff':
        imp_ad = latent_vals[labels == 1].mean(axis=0)
        imp_cn = latent_vals[labels == 0].mean(axis=0)
        importance = imp_ad - imp_cn
    else:
        raise ValueError(f"Modo de pesos SHAP no reconocido: {mode}")

    df = pd.DataFrame({'feature': latent_names, 'importance': importance})
    df['latent_idx'] = (
        df['feature']
        .str.extract(r'(\d+)$', expand=False)
        .astype(int)
    )

    # ordenar por magnitud absoluta (para top_k)
    df = df.reindex(df['importance'].abs().sort_values(ascending=False).index)
    if top_k is not None and top_k > 0:
        df = df.head(min(top_k, len(df)))
    # pesos normalizados (usar magnitud absoluta para normalizar; conservar signo en importance si te interesa)
    denom = df['importance'].abs().sum()
    df['weight'] = 0.0 if denom == 0 else df['importance'] / denom
    return df[['latent_idx', 'weight', 'importance', 'feature']]


def _vae_kwargs_from_args(args: argparse.Namespace, image_size: int) -> Dict[str, Any]:
    return dict(
        input_channels=len(args.channels_to_use),
        latent_dim=args.latent_dim,
        image_size=image_size,
        dropout_rate=args.dropout_rate_vae,
        use_layernorm_fc=getattr(args, 'use_layernorm_vae_fc', False),
        num_conv_layers_encoder=args.num_conv_layers_encoder,
        decoder_type=args.decoder_type,
        intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
        final_activation=args.vae_final_activation,
        num_groups=args.gn_num_groups,
    )


def _load_roi_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == '.joblib':
        return joblib.load(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True).astype(str).tolist()
    # txt/csv
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def generate_saliency_vectorized(vae_model: ConvolutionalVAE,
                                 weights_df: pd.DataFrame,
                                 input_tensor: torch.Tensor,
                                 device: torch.device) -> np.ndarray:
    """Genera saliencia devolviendo (signed, abs), ambos (C,R,R)."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z

    vae_model.eval()
    x = input_tensor.clone().detach().to(device)
    x.requires_grad = True

    # vector de pesos latentes (1, latent_dim)
    w = torch.zeros((1, vae_model.latent_dim), device=device, dtype=torch.float32)
    idx = torch.as_tensor(weights_df['latent_idx'].values, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights_df['weight'].values, device=device, dtype=torch.float32)
    w[0, idx] = vals
    w = w.repeat(x.shape[0], 1)  # expand a batch

    vae_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        # asumiendo forward encode() devuelve (mu, logvar)
        mu, _ = vae_model.encode(x)  # shape (B, latent_dim)
        mu.backward(gradient=w)

    signed, absmap = _grad_to_signed_and_abs(x.grad)
    return signed, absmap


def generate_saliency_smoothgrad(vae_model: ConvolutionalVAE,
                                 weights_df: pd.DataFrame,
                                 input_tensor: torch.Tensor,
                                 device: torch.device,
                                 n_samples: int = 10,
                                 noise_std_perc: float = 0.15) -> np.ndarray:
    """Genera (signed, abs) con SmoothGrad."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z

    input_std = torch.std(input_tensor)
    noise_std = input_std * noise_std_perc
    
    total_signed = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
    total_abs    = np.zeros_like(total_signed)

    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_std
        noisy_input = input_tensor + noise
        s, a = generate_saliency_vectorized(vae_model, weights_df, noisy_input, device)
        total_signed += s
        total_abs    += a

    return total_signed / n_samples, total_abs / n_samples


def generate_saliency_integrated_gradients(vae_model: ConvolutionalVAE,
                                           weights_df: pd.DataFrame,
                                           input_tensor: torch.Tensor,
                                           device: torch.device,
                                           baseline: Optional[torch.Tensor] = None,
                                           n_steps: int = 50) -> np.ndarray:
    """Genera (signed, abs) con Integrated Gradients (Captum)."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z
    if IntegratedGradients is None:
        raise ImportError("Captum no está instalado. No se puede usar 'integrated_gradients'.")

    # 1. Definir el vector de pesos latentes (target para la atribución)
    w = torch.zeros(vae_model.latent_dim, device=device, dtype=torch.float32)
    idx = torch.as_tensor(weights_df['latent_idx'].values, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights_df['weight'].values, device=device, dtype=torch.float32)
    w[idx] = vals

    # 2. Wrapper para Captum: calcula w^T * mu(x)
    def model_forward(x: torch.Tensor) -> torch.Tensor:
        mu, _ = vae_model.encode(x)
        return (mu * w).sum(dim=1)

    # 3. Calcular atribuciones
    ig = IntegratedGradients(model_forward)
    # Baseline:
    #  - si no se pasa, usar cero (comportamiento previo);
    #  - si se pasa (C,R,R) o (1,C,R,R), expandir al batch.
    if baseline is None:
        baselines = torch.zeros_like(input_tensor).to(device)
    else:
        b = baseline.to(device)
        if b.ndim == 3:              # (C,R,R) → (1,C,R,R)
            b = b.unsqueeze(0)
        # expandir a (N,C,R,R) igual que input
        if b.shape[0] == 1 and input_tensor.shape[0] > 1:
            b = b.expand(input_tensor.shape[0], -1, -1, -1)
        baselines = b

    attributions = ig.attribute(input_tensor.to(device), baselines=baselines, n_steps=n_steps)
    A = attributions.mean(dim=0).cpu().numpy()
    signed = _project_to_H(A)
    absmap = _project_to_H(np.abs(A))
    return signed.astype(np.float32), absmap.astype(np.float32)


def _ensure_background_processed(
        background_data: Any,
        preproc: Any,
        feat_names_target: Sequence[str],
        selector: Optional[Any] = None
) -> pd.DataFrame:

    if isinstance(background_data, pd.DataFrame):
        # 1️⃣  Caso ideal: ya vienen con los nombres correctos
        if list(background_data.columns) == list(feat_names_target):
            return background_data

        # 2️⃣  Mismo nº de columnas ⇒ asumimos que YA está procesado,
        #     solo le ponemos los nombres que espera el clasificador
        if background_data.shape[1] == len(feat_names_target):
            log.info("[SHAP] Background ya parece procesado; renombrando columnas.")
            return background_data.set_axis(feat_names_target, axis=1, copy=False)

        # 3️⃣  Si no, intentamos procesarlo desde cero
        log.info("[SHAP] Background DataFrame detectado pero columnas no coinciden; transformando…")
        X_proc = preproc.transform(background_data)
        if selector is not None:
            X_proc = selector.transform(X_proc)
        return pd.DataFrame(X_proc, columns=feat_names_target)
    # -----------------------------------------------------------------
    # resto del cuerpo idéntico, pero añade selector en el branch ndarray
    if isinstance(background_data, np.ndarray):
        if background_data.shape[1] != len(feat_names_target):
            # Este caso es ambiguo: ¿es un array crudo o uno ya procesado con otro
            # preprocesador? El comportamiento más seguro es fallar o, como mínimo,
            # registrar una advertencia severa, ya que no podemos asumir cómo procesarlo.
            raise ValueError(
                f"Background ndarray tiene {background_data.shape[1]} columnas, "
                f"pero se esperaban {len(feat_names_target)}. No se puede continuar de forma segura."
            )
        # Si el número de columnas coincide, lo convertimos a DataFrame.
        return pd.DataFrame(background_data, columns=feat_names_target)
    # -----------------------------------------------------------------
    raise TypeError(f"Tipo de background desconocido: {type(background_data)}")


def _compute_cn_median_baseline(
    *,
    cnad_df: pd.DataFrame,
    tensor_all: np.ndarray,
    channels: Sequence[int],
    norm_params: List[Dict[str, float]],
    test_idx_in_cnad: np.ndarray
) -> torch.Tensor:
    """
    Devuelve un baseline (C,R,R) como la mediana por elemento de los sujetos CN del *TRAIN* del fold,
    en el espacio ya normalizado (mismos params que el VAE del fold).
    """
    # TRAIN del clasificador en este fold (CN/AD): todo menos test
    all_cnad_idx = np.arange(len(cnad_df))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
    train_df = cnad_df.iloc[train_idx_in_cnad]
    cn_train_df = train_df[train_df['ResearchGroup_Mapped'] == 'CN']
    if cn_train_df.empty:
        log.warning("[IG] No hay CN en TRAIN del fold; se intentará con CN del TEST como fallback.")
        cn_train_df = cnad_df.iloc[test_idx_in_cnad][cnad_df.iloc[test_idx_in_cnad]['ResearchGroup_Mapped']=='CN']
        if cn_train_df.empty:
            raise RuntimeError("[IG] No hay CN disponibles en TRAIN ni TEST para construir baseline.")

    gidx = cn_train_df['tensor_idx'].values
    tens = tensor_all[gidx][:, channels, :, :]                  # (N,C,R,R) en escala original
    tens = apply_normalization_params(tens, norm_params)        # normalizado como VAE

    # mediana por elemento → (C,R,R)
    median = np.median(tens, axis=0).astype(np.float32)
    # proyectar a simétrico y hueco, por robustez
    median = 0.5*(median + median.transpose(0,2,1))
    for c in range(median.shape[0]):
        np.fill_diagonal(median[c], 0.0)
    return torch.from_numpy(median).float()                     # (C,R,R)



def _plot_shap_summary(shap_pos: np.ndarray,
                       X_proc_df: pd.DataFrame,
                       out_dir: Path,
                       fold: int,
                       clf: str,
                       base_val: float) -> None:
    # bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_pos, X_proc_df, plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Importancia Global (bar) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_global_importance_bar.png', dpi=150)
    plt.close()

    # beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_pos, X_proc_df, show=False, max_display=20)
    plt.title(f'SHAP Impacto Features (beeswarm) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_beeswarm.png', dpi=150)
    plt.close()

    # waterfall (primer sujeto) --- opcional: proteger si no hay muestras
    if shap_pos.shape[0] > 0:
        exp = shap.Explanation(
            values=shap_pos,
            base_values=np.full(shap_pos.shape[0], base_val, dtype=float),
            data=X_proc_df.values,
            feature_names=list(X_proc_df.columns)
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(exp[0], max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_waterfall_subject_0.png', dpi=150)
        plt.close()



def cmd_saliency(args: argparse.Namespace) -> None:
    # Semillas para reproducibilidad también en la etapa de saliencia
    _set_all_seeds(args.seed)
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    shap_dir = fold_dir / 'interpretability_shap'
    pack_path = shap_dir / f'shap_pack_{args.clf}.joblib'
    if not pack_path.exists():
        raise FileNotFoundError(f"No se encontró shap_pack para {args.clf} en {pack_path}. Corre primero el subcomando 'shap'.")

    pack = joblib.load(pack_path)
    log.info(f"[SALIENCY] fold={args.fold} clf={args.clf}  (pack cargado: {pack_path.name})")
    # ------------------------------------------------------------------
    roi_names: Optional[List[str]] = None
    roi_order_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_joblib.exists():
        roi_names = joblib.load(roi_order_joblib)
        log.info(f"Usando ROI order de {roi_order_joblib}.")
    elif getattr(args, "roi_order_path", None):
        p_roi = Path(args.roi_order_path)
        roi_names = _load_roi_names(p_roi)
        log.info(f"Usando ROI order de --roi_order_path={p_roi}.")

    # Intento con anotaciones si aún no tenemos roi_names
    roi_map_df: Optional[pd.DataFrame] = None
    annot_path = args.roi_annotation_path or args.roi_annotation_csv
    if annot_path is not None:
        try:
            roi_map_df = pd.read_csv(annot_path)
            log.info(f"Cargado fichero de anotaciones: {annot_path}")
            if roi_names is None:
                # ordenar por ROI_TensorIdx si existe; si no, por Original_Index_0_N
                if "ROI_TensorIdx" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("ROI_TensorIdx").reset_index(drop=True)
                elif "Original_Index_0_N" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("Original_Index_0_N").reset_index(drop=True)
                roi_names = roi_map_df["AAL3_Name"].astype(str).tolist()
                log.info("Derivado orden de ROIs desde el CSV de anotaciones.")
        except FileNotFoundError:
            log.error(f"No se pudo leer el fichero de anotaciones: {annot_path}")

    if roi_names is None:
        raise FileNotFoundError(
            "No pude resolver el orden de ROIs. Proporciona --roi_order_path "
            "o asegúrate de tener roi_order_131.joblib o un CSV anotado con AAL3_Name."
        )


    # Pesos latentes ----------------------------------------------------------
    weights_df = get_latent_weights_from_pack(pack, args.shap_weight_mode, args.top_k)
    log.info(f"[SALIENCY] {len(weights_df)} latentes ponderadas. Ejemplo:\n{weights_df.head().to_string(index=False)}")

    # Datos test en espacio de entrada original --------------------------------
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    n_rois_tensor = tens_test.shape[-1]
    if len(roi_names) != n_rois_tensor:
        log.error(f"Número de ROIs ({len(roi_names)}) != dimensión tensor ({n_rois_tensor}). "
                  "Verifica que el orden de ROIs corresponde al tensor usado en entrenamiento.")
        raise ValueError("Desajuste longitud roi_names vs tensor.")

    # VAE ----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tens_test.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)

    labels = np.asarray(pack['test_labels'])  # CN=0 / AD=1 según pack
    x_ad = tens_test_t[labels == 1]
    x_cn = tens_test_t[labels == 0]

    # ------- Construir baseline para IG (si aplica) -------
    ig_baseline_tensor = None

    log.info(f"[SALIENCY] Sujetos AD={x_ad.shape[0]}  CN={x_cn.shape[0]}")

    log.info(f"[SALIENCY] Usando método de saliencia: {args.saliency_method}")

    saliency_fn_map = {
        'vanilla': generate_saliency_vectorized,
        'smoothgrad': lambda v, w, x, d: generate_saliency_smoothgrad(
            v, w, x, d, n_samples=args.sg_n_samples, noise_std_perc=args.sg_noise_std
        ),
        'integrated_gradients': lambda v, w, x, d: generate_saliency_integrated_gradients(
            v, w, x, d, baseline=ig_baseline_tensor, n_steps=args.ig_n_steps
        ),
    }
    saliency_fn = saliency_fn_map[args.saliency_method]

    out_dir = fold_dir / f"interpretability_{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # si el método es IG, resolver baseline según el flag
    if args.saliency_method == 'integrated_gradients':
        if args.ig_baseline == 'zeros':
            log.info("[IG] Baseline: tensor de ceros (mismo espacio normalizado).")
            ig_baseline_tensor = None
        elif args.ig_baseline == 'cn_median_test':
            log.info("[IG] Baseline: mediana CN del TEST del fold (mismo espacio normalizado).")
            if x_cn.shape[0] == 0:
                raise RuntimeError("[IG] No hay CN en el TEST para baseline 'cn_median_test'. Usa 'cn_median_train'.")
            cn_med = torch.median(x_cn, dim=0).values.detach().cpu()  # (C,R,R)
            ig_baseline_tensor = cn_med
        elif args.ig_baseline == 'cn_median_train':
            log.info("[IG] Baseline: mediana CN del TRAIN del fold (sin fuga de test).")
            ig_baseline_tensor = _compute_cn_median_baseline(
                cnad_df=cnad,
                tensor_all=tensor_all,
                channels=args.channels_to_use,
                norm_params=norm_params,
                test_idx_in_cnad=test_idx_in_cnad
            )
        np.save(out_dir / f"ig_baseline_{args.ig_baseline}.npy",
                (ig_baseline_tensor.cpu().numpy() if ig_baseline_tensor is not None else np.zeros_like(tens_test_t[0].cpu().numpy())))

    sal_ad_signed, sal_ad_abs = saliency_fn(vae, weights_df, x_ad, device)
    sal_cn_signed, sal_cn_abs = saliency_fn(vae, weights_df, x_cn, device)
    sal_diff_signed = sal_ad_signed - sal_cn_signed
    # diferencia en sentido signed; la variante abs se define como |diff_signed|
    sal_diff_abs = np.abs(sal_diff_signed)

    # Guardar mapas ------------------------------------------------------------
    out_dir = fold_dir / f"interpretability_{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "" if args.saliency_method == "vanilla" else f"_{args.saliency_method}"
    file_suffix = f"{method_tag}_top{args.top_k}"

    # Grabar por grupo y diferencial (signed y abs)
    np.save(out_dir / f"saliency_map_ad_signed{file_suffix}.npy",  sal_ad_signed)
    np.save(out_dir / f"saliency_map_ad_abs{file_suffix}.npy",     sal_ad_abs)
    np.save(out_dir / f"saliency_map_cn_signed{file_suffix}.npy",  sal_cn_signed)
    np.save(out_dir / f"saliency_map_cn_abs{file_suffix}.npy",     sal_cn_abs)
    np.save(out_dir / f"saliency_map_diff_signed{file_suffix}.npy", sal_diff_signed)
    np.save(out_dir / f"saliency_map_diff_abs{file_suffix}.npy",    sal_diff_abs)
 

    _ranking_and_heatmap(
        saliency_map_diff_signed=sal_diff_signed,
        saliency_map_diff_abs=sal_diff_abs,
        roi_map_df=roi_map_df,
        roi_names=roi_names,
        out_dir=out_dir,
        fold=args.fold,
        clf=args.clf,
        top_k=args.top_k,
        method_tag=method_tag
    )

    # ===== NEW =====
    # (1) Contribución por canal desde el mapa diferencial
    l1_abs = sal_diff_abs.sum(axis=(1,2))  # L1 (abs)
    l1_sgn = np.sign(sal_diff_signed).sum(axis=(1,2))  # suma de signos (diagnóstico)
    frac_abs = l1_abs / (l1_abs.sum() + 1e-12)          # fracción relativa
    # nombres de canales (si el usuario los pasa, usarlos; si no, usar índices)
    ch_used = list(args.channels_to_use)
    if getattr(args, "channel_names", None) and len(args.channel_names) == len(ch_used):
        ch_names = list(args.channel_names)
    else:
        ch_names = [f"Ch{c}" for c in ch_used]
    # guardar con nombres de columnas amigables para el notebook
    chan_df = pd.DataFrame({
        'channel_index_used': ch_used,
        'channel_name': ch_names,
        'l1_norm_abs': l1_abs,
        'l1_norm_fraction_abs': frac_abs,
        'signed_sum': l1_sgn
    })
    # Guardar con y sin sufijo para que el notebook lo encuentre directo
    chan_csv_suff = out_dir / f'channel_contributions{file_suffix}.csv'
    chan_csv_nosuff = out_dir / 'channel_contributions.csv'
    chan_df.to_csv(chan_csv_suff, index=False)
    chan_df.to_csv(chan_csv_nosuff, index=False)

    plt.figure(figsize=(6,4)); plt.bar(np.arange(len(l1_abs)), frac_abs)
    plt.xlabel('Channel'); plt.ylabel('Fraction of total |ΔSal|')
    plt.title(f'Channel contributions – fold {args.fold}'); plt.tight_layout()
    plt.savefig(out_dir / f'channel_contributions{file_suffix}.png', dpi=150)
    plt.savefig(out_dir / 'channel_contributions.png', dpi=150)
    plt.close()

    # (2) Network-pair matrices and quick visuals from the annotated ranking
    edge_csv = out_dir / f"ranking_conexiones_ANOTADO{file_suffix}.csv"
    if edge_csv.exists():
        df_edges = pd.read_csv(edge_csv)
        # choose network columns available
        net_src_col = 'src_Refined_Network' if 'src_Refined_Network' in df_edges.columns else 'src_Yeo17_Network'
        net_dst_col = 'dst_Refined_Network' if 'dst_Refined_Network' in df_edges.columns else 'dst_Yeo17_Network'
        nets = sorted(set(df_edges[net_src_col].astype(str)) | set(df_edges[net_dst_col].astype(str)))
        pos = {n:i for i,n in enumerate(nets)}
        M_abs = np.zeros((len(nets),len(nets)), float)
        M_sgn = np.zeros_like(M_abs)
        for _, r in df_edges.iterrows():
            i = pos[str(r[net_src_col])]; j = pos[str(r[net_dst_col])]
            v = float(r['Saliency_Signed']); a = float(r['Saliency_Abs'])
            M_abs[i,j]+=a; M_abs[j,i]+=a; M_sgn[i,j]+=v; M_sgn[j,i]+=v
        pd.DataFrame(M_abs, index=nets, columns=nets).to_csv(out_dir / f'network_pairs_sumabs{file_suffix}.csv', index=True)
        pd.DataFrame(M_sgn, index=nets, columns=nets).to_csv(out_dir / f'network_pairs_signed{file_suffix}.csv', index=True)
        # visuals (default colormap)
        plt.figure(figsize=(8,7)); plt.imshow(M_abs)
        plt.xticks(range(len(nets)), nets, rotation=90); plt.yticks(range(len(nets)), nets)
        plt.title('Network-pair energy (sum |ΔSal|)'); plt.colorbar()
        plt.tight_layout(); plt.savefig(out_dir / f'heatmap_network_pairs_sumabs{file_suffix}.png', dpi=150); plt.close()
        plt.figure(figsize=(8,7)); plt.imshow(M_sgn)
        plt.xticks(range(len(nets)), nets, rotation=90); plt.yticks(range(len(nets)), nets)
        plt.title('Network-pair signed (ΔSal, AD>CN +)'); plt.colorbar()
        plt.tight_layout(); plt.savefig(out_dir / f'heatmap_network_pairs_signed{file_suffix}.png', dpi=150); plt.close()

    # (3) Robust hubs (degree-controlled node-strength) for K in {50,100,200}
    if edge_csv.exists():
        df_edges = pd.read_csv(edge_csv)
        for K in (50,100,200):
            sub = df_edges[df_edges['Rank']<=K].copy()
            nodes = sorted(set(sub['src_AAL3_Name'].astype(str)) | set(sub['dst_AAL3_Name'].astype(str)))
            deg = {n:0 for n in nodes}; strg = {n:0.0 for n in nodes}
            for _, r in sub.iterrows():
                a = str(r['src_AAL3_Name']); b = str(r['dst_AAL3_Name'])
                # usar Saliency_Abs si existe; si no, caer a |Saliency_Signed|
                col = 'Saliency_Abs' if 'Saliency_Abs' in r else 'Saliency_Signed'
                w = float(abs(r[col]))
                deg[a]+=1; deg[b]+=1; strg[a]+=w; strg[b]+=w
            # Construir tabla y residualizar fuerza respecto a grado
            tab = pd.DataFrame({
                'node': nodes,
                'degree': [deg[n] for n in nodes],
                'strength': [strg[n] for n in nodes]
            })
            if tab['degree'].max() > 0:
                x = tab['degree'].to_numpy(dtype=float)
                y = tab['strength'].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                tab['residual_strength'] = y - (intercept + slope * x)
            else:
                tab['residual_strength'] = 0.0
            tab.sort_values('residual_strength', ascending=False).to_csv(out_dir / f'node_robust_hubs_top{K}{file_suffix}.csv', index=False)
    # ===== END NEW =====

 
    # Guardar args usados ------------------------------------------------------
    with open(out_dir / f"run_args_saliency{file_suffix}.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"[SALIENCY] Completo. Resultados en {out_dir}")


# ---------------------------------------------------------------------------
# Ranking + visualización
# ---------------------------------------------------------------------------

def _ranking_and_heatmap(saliency_map_diff_signed: np.ndarray,
                         saliency_map_diff_abs: np.ndarray,
                         roi_map_df: Optional[pd.DataFrame],
                         roi_names: Sequence[str],
                         out_dir: Path,
                         fold: int,
                         clf: str,
                         top_k: int,
                         annot_df: Optional[pd.DataFrame] = None,
                         method_tag: str = "") -> None:
    # Ambos (C,R,R) → promediamos sobre canales
    sal_m_sgn = saliency_map_diff_signed.mean(axis=0)
    sal_m_abs = saliency_map_diff_abs.mean(axis=0)
    n_rois = sal_m_sgn.shape[0]
    # Crear la tabla de conexiones con índices numéricos
    ut_indices = np.triu_indices(n_rois, k=1)
    df_edges = pd.DataFrame({
        'idx_i': ut_indices[0],
        'idx_j': ut_indices[1],
        'Saliency_Signed': sal_m_sgn[ut_indices],
        'Saliency_Abs':    sal_m_abs[ut_indices]
    })
    # <NUEVO> Anotar el DataFrame de conexiones si el mapa de ROIs está disponible
    if roi_map_df is not None:
        # Alinear al orden del tensor
        if 'ROI_TensorIdx' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('ROI_TensorIdx').reset_index(drop=True)
            idx_col = 'ROI_TensorIdx'
        elif 'Original_Index_0_N' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('Original_Index_0_N').reset_index(drop=True)
            idx_col = 'Original_Index_0_N'
        else:
            log.warning("roi_map_df no tiene columnas de índice reconocibles; se omitirá la anotación.")
            idx_col = None

        if idx_col is not None:
            # Diccionarios de anotación
            name_map   = roi_map_df.set_index(idx_col)['AAL3_Name'].astype(str).to_dict()
            lobe_map   = roi_map_df.set_index(idx_col)['Macro_Lobe'].astype(str).to_dict()           if 'Macro_Lobe' in roi_map_df else {}
            net_map    = roi_map_df.set_index(idx_col)['Refined_Network'].astype(str).to_dict()      if 'Refined_Network' in roi_map_df else {}
            y17_map    = roi_map_df.set_index(idx_col)['Yeo17_Network'].astype(str).to_dict()        if 'Yeo17_Network' in roi_map_df else {}

            # Añadir columnas al ranking
            df_edges['ROI_i_name'] = df_edges['idx_i'].map(name_map)
            df_edges['ROI_j_name'] = df_edges['idx_j'].map(name_map)
            df_edges['src_AAL3_Name'] = df_edges['ROI_i_name']
            df_edges['dst_AAL3_Name'] = df_edges['ROI_j_name']
            df_edges['src_Macro_Lobe'] = df_edges['idx_i'].map(lobe_map)
            df_edges['dst_Macro_Lobe'] = df_edges['idx_j'].map(lobe_map)
            df_edges['src_Refined_Network'] = df_edges['idx_i'].map(net_map)
            df_edges['dst_Refined_Network'] = df_edges['idx_j'].map(net_map)
            df_edges['src_Yeo17_Network'] = df_edges['idx_i'].map(y17_map)
            df_edges['dst_Yeo17_Network'] = df_edges['idx_j'].map(y17_map)      


    df_edges = df_edges.sort_values('Saliency_Abs', ascending=False)
    df_edges.insert(0, 'Rank', range(1, len(df_edges) + 1))
    file_suffix = f"{method_tag}_top{top_k}"
    edge_csv_path = out_dir / f"ranking_conexiones_ANOTADO{file_suffix}.csv"
    df_edges.to_csv(edge_csv_path, index=False)
    log.info(f"[SALIENCY] Ranking de conexiones ANOTADO guardado: {edge_csv_path}")

    if annot_df is not None:
        meta = annot_df[['AAL3_Name','Macro_Lobe','Refined_Network']]
        df_edges = (df_edges
                    .merge(meta, left_on='ROI_i_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_i','Refined_Network':'Network_i'})
                    .drop(columns='AAL3_Name')
                    .merge(meta, left_on='ROI_j_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_j','Refined_Network':'Network_j'})
                    .drop(columns='AAL3_Name'))
    # preview top 20
    preview_cols = [
        'Rank', 'src_AAL3_Name', 'dst_AAL3_Name', 'Saliency_Signed', 'Saliency_Abs',
        'src_Refined_Network', 'dst_Refined_Network'
    ]
    preview_cols_exist = [c for c in preview_cols if c in df_edges.columns]
    log.info("Top 20 conexiones anotadas:\n" + df_edges.head(20)[preview_cols_exist].to_string())
 
 

    # heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sal_m_sgn, cmap='coolwarm', center=0,
                xticklabels=list(roi_names)[:n_rois], yticklabels=list(roi_names)[:n_rois],
                cbar_kws={'label': 'Saliencia Diferencial (AD > CN)'} )
    plt.title(f'Mapa de Saliencia Diferencial (AD vs CN) - Fold {fold} - {clf.upper()}{method_tag.replace("_", " ").title()}')
    plt.tight_layout(); plt.savefig(out_dir / f"mapa_saliencia_diferencial{file_suffix}.png", dpi=150); plt.close()


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--roi_annotation_csv', default=None,
               help='CSV con anotaciones de ROI (Macro_Lobe, Redes, etc.).')
    p.add_argument('--run_dir', required=True, help='Directorio raíz del experimento (donde viven fold_*).')
    p.add_argument('--fold', type=int, required=True, help='Fold a analizar (1-indexed).')
    p.add_argument('--clf', required=True, help='Clasificador (xgb, svm, logreg, gb, rf, ...).')
    p.add_argument('--global_tensor_path', required=True, help='Ruta al GLOBAL_TENSOR .npz usado en entrenamiento.')
    p.add_argument('--metadata_path', required=True, help='Ruta al CSV de metadatos usado en entrenamiento.')
    p.add_argument('--channels_to_use', type=int, nargs='*', required=True, help='Índices de canales usados en entrenamiento.')
    p.add_argument('--latent_dim', type=int, required=True, help='Dimensión latente del VAE.')
    p.add_argument('--latent_features_type', choices=['mu','z'], default='mu', help='Usar mu o z como features latentes.')
    p.add_argument('--metadata_features', nargs='*', default=None, help='Columnas de metadatos añadidas al clasificador.')
    # Arquitectura VAE
    p.add_argument('--seed', type=int, default=42, help='Semilla global para numpy/torch/shap.')
    p.add_argument('--num_conv_layers_encoder', type=int, default=4)
    p.add_argument('--decoder_type', default='convtranspose', choices=['convtranspose','upsample_conv'])
    p.add_argument('--dropout_rate_vae', type=float, default=0.2)
    p.add_argument('--use_layernorm_vae_fc', action='store_true')
    p.add_argument('--intermediate_fc_dim_vae', default='quarter')
    p.add_argument('--vae_final_activation', default='tanh', choices=['tanh','sigmoid','linear'])
    p.add_argument('--gn_num_groups', type=int, default=16, help='n grupos para GroupNorm en VAE.')
    p.add_argument('--channel_names', nargs='*', default=None,
                   help='(Opcional) nombres legibles de los canales, longitud = len(channels_to_use).')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pipeline Unificado de Interpretabilidad (VAE+Clasificador).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # subcomando SHAP ---------------------------------------------------------
    p_shap = sub.add_parser('shap', help='Calcular y guardar valores SHAP para un fold+clf.')
    _add_shared_args(p_shap)

    p_shap.add_argument('--roi_order_path', default=None, help='(Opcional) ruta a ROI order si no está en run_dir.')
    p_shap.add_argument('--kernel_nsamples', type=int, default=100, help='nsamples para KernelExplainer (modelos no tree).')

    # subcomando SALIENCY -----------------------------------------------------
    p_sal = sub.add_parser('saliency', help='Generar mapas de saliencia a partir del shap_pack.')
    _add_shared_args(p_sal)
    p_sal.add_argument('--roi_order_path', default=None,
                    help='(Opcional) ruta a ROI order (.joblib/.npy/.txt) si no está en run_dir.')
    p_sal.add_argument('--roi_annotation_path', required=True, help='Ruta al fichero maestro de anotaciones de ROIs (roi_info_master.csv).')
    p_sal.add_argument('--top_k', type=int, default=50, help='Nº máx features latentes a usar.')
    p_sal.add_argument('--shap_weight_mode', default='ad_vs_cn_diff', choices=['mean_abs','mean_signed','ad_vs_cn_diff'],
                       help='Cómo convertir valores SHAP latentes en pesos para saliencia.')
    p_sal.add_argument('--saliency_method', default='vanilla', choices=['vanilla', 'smoothgrad', 'integrated_gradients'],
                       help='Método para generar el mapa de saliencia.')
    # Args para SmoothGrad
    p_sal.add_argument('--sg_n_samples', type=int, default=10, help='[SmoothGrad] Nº de muestras con ruido a promediar.')
    p_sal.add_argument('--sg_noise_std', type=float, default=0.15, help='[SmoothGrad] Desv. estándar del ruido como % de la desv. estándar de la entrada.')
    # Args para Integrated Gradients
    p_sal.add_argument('--ig_n_steps', type=int, default=50, help='[IG] Nº de pasos para la aproximación de la integral.')
    
    p_sal.add_argument('--ig_baseline',
                       default='cn_median_train',
                       choices=['zeros', 'cn_median_train', 'cn_median_test'],
                       help='Baseline para IG: ceros (antiguo), mediana CN del TRAIN (recomendado), o mediana CN del TEST.')
 

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # Semillas globales también aquí (por si algún camino llama antes)
    if hasattr(args, 'seed'):
        _set_all_seeds(int(args.seed))
    if args.cmd == 'shap':
        cmd_shap(args)
    elif args.cmd == 'saliency':
        cmd_saliency(args)
    else:
        raise ValueError(f"Subcomando desconocido: {args.cmd}")


if __name__ == '__main__':
    main()
