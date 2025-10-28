#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers7.py
===============

Módulo centralizado para definir clasificadores de scikit-learn y sus grids de
búsqueda de hiperparámetros, encapsulados en un pipeline robusto con pre-procesado.

Versión: 3.2.1 - Espacios de búsqueda refinados
"""
from __future__ import annotations
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)


import torch
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Any, Dict, List, Tuple
import os
import lightgbm
# ajustar el nivel de verbosidad globalmente
# ajustar el nivel de verbosidad globalmente (si está disponible)
if hasattr(lightgbm, "set_config"):
    lightgbm.set_config(verbosity=1)



logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

for noisy in ["lightgbm", "optuna", "sklearn", "xgboost"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)



try:
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    has_gpu = False

print("¿GPU visible?:", has_gpu)

os.environ["XGB_HIDE_LOG"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from lightgbm.basic import _LIB, _safe_call
# Algunos builds de LightGBM no tienen LGBM_SetLogLevel en _LIB:
if hasattr(_LIB, "LGBM_SetLogLevel"):
    try:
        _safe_call(_LIB.LGBM_SetLogLevel(1))
    except Exception:
        # falla silenciosamente si no existe o da otro error
        pass

# 0 = fatal, 1 = error, 2 = warning, 3 = info, 4 = debug
#_safe_call(_LIB.LGBM_SetLogLevel(1))


ClassifierPipelineAndGrid = Tuple[ImblearnPipeline, Dict[str, Any], int]

# Detectar GPU y cuML
try:
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    has_gpu = False

# Intentar importar cuML (opcional)
try:
    from cuml.linear_model import LogisticRegression as cuLogistic
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.preprocessing import SMOTE as cuSMOTE
    has_cuml = True
    logger.info("[cuML] ➜ disponible: aceleración GPU para logreg, SVM, RF, SMOTE")
except ImportError:
    has_cuml = False
    logger.info("[cuML] ➜ no disponible")

if hasattr(_LIB, "LGBM_SetLogLevel"):
    try:
        _safe_call(_LIB.LGBM_SetLogLevel(0))   # 0 = FATAL
    except Exception:
        pass



def get_available_classifiers() -> List[str]:
    """Devuelve la lista de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb", "cat"]

def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    """Convierte un string '128,64' a una tupla (128, 64)."""
    if not hidden_layers_str:
        return (128, 64)
    return tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())

def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False,
    use_feature_selection: bool = False
) -> ClassifierPipelineAndGrid:
    """
    Construye un pipeline de imblearn y devuelve el pipeline, el grid de parámetros y el n_iter.
    """
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")

    class_weight = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any]
    n_iter_search = 150

    if ctype == 'svm':
        model = SVC(probability=False, random_state=seed, class_weight=class_weight, cache_size=500)
        param_distributions = {
            'model__C': FloatDistribution(1, 1e4, log=True), 
            'model__gamma': FloatDistribution(1e-7, 1e-3, log=True),
            'model__kernel': CategoricalDistribution(['rbf']),
        }
        n_iter_search = 200

    elif ctype == 'logreg':
        model = LogisticRegression(random_state=seed, class_weight=class_weight, solver='liblinear', max_iter=2000)
        param_distributions = {
            'model__C': FloatDistribution(1e-5, 1, log=True)
        }
        n_iter_search = 200

    elif ctype == "gb":
        # --- 1. Instancia base ---
        model = LGBMClassifier(
            random_state=seed,
            objective="binary",
            class_weight=class_weight,
            n_jobs=1,          # toda la paralelización la lleva Optuna
            verbose=-1
        )

        # --- 2. Soporte GPU limpio ---
        if has_gpu:
            try:
                if bool(_safe_call(_LIB.LGBM_HasGPU())):
                    model.set_params(device_type="gpu", gpu_use_dp=True)
                    print("[LightGBM] ➜ GPU activada")
                else:
                    model.set_params(device_type="cpu")
                    print("[LightGBM] ⚠ Build sin GPU, usando CPU")
            except Exception:
                model.set_params(device_type="cpu")
                print("[LightGBM] ⚠ No se pudo comprobar la GPU, usando CPU")

        # --- 3. Espacio de búsqueda Optuna ---
        param_distributions = {
            #"model__boosting_type": CategoricalDistribution(["dart", "goss"]),
            # Profundidad y tamaño del árbol
            "model__max_depth":       IntDistribution(3, 12),
            "model__num_leaves":      IntDistribution(8, 2**10),   # coherente con max_depth

            # Muestras y features por árbol
            "model__bagging_fraction":FloatDistribution(0.5, 1.0),
            "model__feature_fraction":FloatDistribution(0.5, 1.0),
            "model__bagging_freq":    IntDistribution(1, 10),

            # Aprendizaje
            "model__learning_rate":   FloatDistribution(5e-4, 0.01, log=True),
            "model__n_estimators":    IntDistribution(300, 1000),

            # Regularización
            "model__min_child_samples":IntDistribution(5, 50),
            "model__min_child_weight": FloatDistribution(1e-3, 10, log=True),
            "model__min_split_gain":   FloatDistribution(0.0, 1.0),
            "model__reg_alpha":        FloatDistribution(1e-3, 1.0, log=True),
            "model__reg_lambda":       FloatDistribution(1e-3, 1.0, log=True),
        }

        # --- 4. Número de iteraciones dinámico ---
        n_param       = len(param_distributions)
        n_iter_search = int(round((15 * n_param) / 10.0)) * 10  # múltiplo de 10

    elif ctype == 'rf':
        print("[RandomForest] ➜ Usando implementación de scikit-learn (CPU).")
        model = RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=-1)
        param_distributions = {
            'model__n_estimators': IntDistribution(100, 1200),
            'model__max_features': CategoricalDistribution(['sqrt', 'log2', 0.2, 0.4]),
            'model__max_depth': IntDistribution(8, 50),
            'model__min_samples_split': IntDistribution(2, 30), 
            'model__min_samples_leaf': IntDistribution(1, 20)
        }
        n_iter_search = 150 # Restaurado a un valor más robusto
    
    elif ctype == 'mlp':
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True, n_iter_no_change=25)
        param_distributions = {
            'model__alpha': FloatDistribution(1e-5, 1e-1, log=True),
            'model__learning_rate_init': FloatDistribution(1e-5, 1e-2, log=True),
        }
        n_iter_search = 200

    elif ctype == "xgb":
        model = XGBClassifier(random_state=seed, eval_metric="auc", n_jobs=1, tree_method="hist", device="cuda", verbosity=0)
        if has_gpu:
            print("[XGBoost] ➜  Se usará GPU (device=cuda)")
        else:
            print("[XGBoost] ⚠  GPU no disponible, usando CPU.")
        param_distributions = {
            "model__gamma": FloatDistribution(0.0, 5.0),
            "model__n_estimators": IntDistribution(500, 1500), # Rango ampliado
            "model__learning_rate": FloatDistribution(1e-4, 0.1, log=True),
            "model__max_depth": IntDistribution(4, 12), # Rango ampliado
            "model__subsample": FloatDistribution(0.3, 1.0), # Corregido a 1.0
            "model__colsample_bytree": FloatDistribution(0.5, 1.0), # Corregido a 1.0
            "model__min_child_weight": FloatDistribution(0.5, 10., log=True), # Rango ampliado
        }
        n_iter_search = 200 # Aumentado para el espacio más grande

    elif ctype == "cat":
        model = CatBoostClassifier(random_state=seed, eval_metric="Logloss", verbose=0, loss_function="Logloss", thread_count=1)
        if has_gpu:
            model.set_params(task_type="GPU", devices="0:0")
            print("[CatBoost] ➜  Se usará GPU")
        else:
            print("[CatBoost] ⚠  GPU no disponible, usando CPU.")
        param_distributions = {
            "model__depth": IntDistribution(4, 8),
            "model__learning_rate": FloatDistribution(1e-3, 0.08, log=True),
            "model__l2_leaf_reg": FloatDistribution(0.1, 20.0, log=True),
            "model__iterations": IntDistribution(400, 1500),
            "model__bagging_temperature": FloatDistribution(0.1, 0.9),
        }
        n_iter_search = 120

    if calibrate and ctype in ["svm", "gb", "rf"]:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        _cal = CalibratedClassifierCV(model.model if hasattr(model, "model") else model)
        _inner = "estimator" if "estimator" in _cal.get_params() else "base_estimator"
        param_distributions = { f"model__{_inner}__{k.split('__', 1)[1]}": v for k, v in param_distributions.items() }

    # ---------- 1.  Escalador ----------
    scaler_step = ('scaler', 'passthrough') if ctype in ['rf','gb','xgb','cat'] \
                                          else ('scaler', StandardScaler())

    # ---------- 2.  Oversampler ----------
    oversampler_step: tuple | None = None
    if use_smote:
        oversampler_step = ('smote', SMOTE(random_state=seed))
        logger.info(f"[SMOTE] ➜ aplicado sólo dentro de folds (imblearn Pipeline).")
        if tune_sampler_params:
            param_distributions['smote__k_neighbors'] = IntDistribution(3, 25)

    # ---------- 3. Feature Selector (Opcional) ----------
    feature_selector_step: tuple | None = None
    if use_feature_selection:
        feature_selector_step = ('feature_selector', SelectKBest(f_classif))
        # Permitir que Optuna busque el número óptimo de features
        # El límite superior depende de latent_dim + metadata_features, pero lo ponemos genérico
        param_distributions['feature_selector__k'] = IntDistribution(20, 256)
        logger.info("[SelectKBest] ➜ añadido al pipeline con 'k' tunable (20-256).")

    # ---------- 4.  Modelo ----------
    model_step = ('model', model)

    # ---------- 5.  Construcción ordenada ----------
    #steps_ordered = [scaler_step] + ([oversampler_step] if oversampler_step else []) + [model_step]
    steps_ordered = ([scaler_step] + ([feature_selector_step] if feature_selector_step else []) + ([oversampler_step] if oversampler_step else []) + [model_step])
    full_pipeline = ImblearnPipeline(steps=steps_ordered)
    return full_pipeline, param_distributions, n_iter_search