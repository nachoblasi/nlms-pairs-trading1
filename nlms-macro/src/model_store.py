"""
model_store.py — Persistencia del modelo mu_predictor entre runs.

La red neuronal mejora con el uso: cada par que entrenas aporta nuevos
ejemplos (X, Y) al dataset acumulado. En el siguiente run, el modelo se
re-entrena sobre TODOS los datos históricos de TODOS los pares vistos.

Flujo:
    Run 1 (V/MA):
        load_accumulated_data() → vacío
        entrena con solo V/MA (~9.500 muestras)
        save_accumulated_data(X_vma, Y_vma)

    Run 2 (GS/MS):
        load_accumulated_data() → (X_vma, Y_vma)
        en cada fold: train_mu_predictor(..., X_extra=X_vma, Y_extra=Y_vma)
        → el modelo ve GS/MS + V/MA de fondo → más robusto
        save_accumulated_data(stack(X_vma, X_gsms), ...)

    Run 3 (KO/PEP):
        load_accumulated_data() → ~19.000 muestras de V/MA + GS/MS
        ...

Las features están normalizadas por el nivel de precio → son comparables
entre pares con distintas escalas de precio.

Archivos creados:
    models/training_data.npz  — dataset acumulado (X, Y)
    models/mu_predictor.pkl   — (model, scaler) del último entrenamiento completo
"""

import numpy as np
import joblib
from pathlib import Path

_MODELS_DIR = Path("models")
_DATA_PATH  = _MODELS_DIR / "training_data.npz"
_MODEL_PATH = _MODELS_DIR / "mu_predictor.pkl"


def load_accumulated_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Carga el dataset acumulado de runs anteriores.

    Retorna (X, Y) donde:
        X : (n_samples, n_features) — features normalizadas por precio
        Y : (n_samples,)            — targets μ
    Si no hay datos previos, retorna arrays vacíos.
    """
    if _DATA_PATH.exists():
        data = np.load(_DATA_PATH)
        return data["X"], data["Y"]
    return np.empty((0, 0)), np.empty(0)


def save_accumulated_data(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Guarda el dataset acumulado en disco.

    Sobreescribe el archivo anterior con el nuevo dataset combinado
    (datos previos + nuevos del run actual).
    """
    _MODELS_DIR.mkdir(exist_ok=True)
    np.savez(_DATA_PATH, X=X, Y=Y)


def save_model(model, scaler) -> None:
    """Guarda (model, scaler) del último entrenamiento en disco."""
    _MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump((model, scaler), _MODEL_PATH)


def load_model():
    """
    Carga (model, scaler) del último entrenamiento.
    Retorna (None, None) si no hay modelo guardado.
    """
    if _MODEL_PATH.exists():
        return joblib.load(_MODEL_PATH)
    return None, None


def reset() -> None:
    """Borra el dataset acumulado y el modelo guardado (empezar de cero)."""
    deleted = []
    for path in [_DATA_PATH, _MODEL_PATH]:
        if path.exists():
            path.unlink()
            deleted.append(path.name)
    if deleted:
        print(f"    Eliminado: {', '.join(deleted)}")
    else:
        print("    No había datos que eliminar.")


def info() -> str:
    """Devuelve un resumen del estado del modelo persistido."""
    if not _DATA_PATH.exists():
        return "Sin datos acumulados (primera ejecución o --reset reciente)."
    data = np.load(_DATA_PATH)
    n, d = data["X"].shape
    return (f"{n:,} muestras acumuladas  |  {d} features  "
            f"|  [{_DATA_PATH}]")
