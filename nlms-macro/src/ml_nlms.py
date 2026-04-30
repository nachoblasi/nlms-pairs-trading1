"""
ml_nlms.py — Filtro NLMS con step-size μ controlado por Machine Learning.

Idea central
────────────
El VS-NLMS estándar adapta μ usando una heurística matemática:
    μ[t] = clip(α·μ[t-1] + γ·sign(e[t])·sign(e[t-1]), μ_min, μ_max)

Esto funciona bien, pero la heurística es fija y no aprende de los datos.

Este módulo reemplaza esa heurística por un MLPRegressor entrenado sobre
los datos del periodo de TRAIN del walk-forward:

    ENTRENAMIENTO:
        - Correr N filtros NLMS en paralelo, cada uno con μ fijo distinto
          (ej: 0.01, 0.05, 0.1, 0.2).
        - En cada instante t: el μ "ganador" es el del filtro con menor |e[t]|.
          Ese valor es el TARGET Y[t] para la red.
        - Las FEATURES X[t] son el historial reciente de errores absolutos:
          [|e[t-1]|, |e[t-2]|, ..., |e[t-W]|, std(|e|), sign_corr]
        - Entrenar MLPRegressor(X, Y) → aprende a mapear "patrón de errores → μ óptimo".

    EJECUCIÓN (ML_VSNLMSFilter):
        - En cada update(x, y), el filtro extrae sus propias features (historial de errores)
        - Las pasa al modelo → predice μ[t]
        - Clip a [μ_min, μ_max] por seguridad
        - Actualiza w con la ecuación NLMS estándar usando ese μ predicho

La ecuación de actualización NO cambia respecto al NLMS clásico:
    w[t+1] = w[t] + μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)

Solo cambia cómo se calcula μ[t]: en vez de la heurística Kwong-Johnston,
la red neuronal lo predice a partir del historial de errores.

Reversibilidad
──────────────
Este archivo es completamente independiente del resto del proyecto.
Para eliminarlo: borrar src/ml_nlms.py y walk_forward_ml_mu.py.
No modifica ningún archivo existente.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ── Constantes ────────────────────────────────────────────────────────────────

# μ candidatos para la competición de filtros paralelos.
# Rango amplio: cubre desde muy conservador (0.005) hasta muy agresivo (0.5).
# IMPORTANTE: incluir 0.4 y 0.5 porque el VS-NLMS baseline opera en μ≈0.43 de media
# (Kwong-Johnston empuja μ alto por autocorrelación positiva de errores).
# Si los candidatos no cubren ese rango, el modelo aprende a recomendar siempre μ bajo.
DEFAULT_MU_CANDIDATES = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Número de errores pasados usados como features.
# 5 errores ≈ 1 semana de trading. Suficiente para capturar tendencias
# sin introducir demasiada dimensionalidad.
DEFAULT_FEATURE_WINDOW = 5

# Ventana (en días) para evaluar la mean-reversion del spread en cada paso.
# 20 días ≈ 1 mes de trading. Equilibrio entre estabilidad estadística y reactividad.
DEFAULT_TARGET_WINDOW = 20

# Rango seguro para μ (igual que en VSNLMSFilter).
MU_MIN_DEFAULT = 0.001
MU_MAX_DEFAULT = 0.5


# ── Generación de datos de entrenamiento ──────────────────────────────────────

def train_mu_predictor(
    price_x: np.ndarray,
    price_y: np.ndarray,
    mu_candidates: list = DEFAULT_MU_CANDIDATES,
    feature_window: int = DEFAULT_FEATURE_WINDOW,
    target_window: int = DEFAULT_TARGET_WINDOW,
    eps: float = 1e-8,
    X_extra: np.ndarray = None,
    Y_extra: np.ndarray = None,
) -> tuple:
    """
    Genera datos de entrenamiento (X, Y) corriendo N filtros NLMS en paralelo
    y entrena un MLPRegressor que predice el μ óptimo a partir del historial de errores.

    ──────────────────────────────────────────────────────────────────────────
    Lógica de la "competición de filtros" (target_mode = mean_reversion):

        Para cada instante t, tenemos N filtros corriendo simultáneamente.
        Cada filtro_i tiene su propio peso w_i[t] actualizado siempre con su μ_i fijo.

        En cada paso t (con t >= target_window):
            Se evalúa la calidad del spread de cada filtro sobre la ventana
            [t - target_window, t) como su autocorrelación lag-1.

            Autocorrelación lag-1 < 0 → spread mean-reverting → bueno para trading
            Autocorrelación lag-1 > 0 → spread con tendencia  → malo para trading

            El "ganador" en t es el filtro con la autocorrelación más NEGATIVA.
            Target Y[t] = μ del filtro ganador.

    ──────────────────────────────────────────────────────────────────────────
    ¿Por qué mean-reversion en lugar de mínimo |e[t]|?

        La versión original usaba argmin |e[t]| como target. Esto es incorrecto
        para pairs trading porque:

        - Un μ pequeño produce β lento (hedge ratio estable pero impreciso).
          → Spread = e[t] tiene varianza baja (parece "bueno" en mínimo |e|).
          → Pero el spread no revierte porque β no sigue el hedge ratio real.
          → Trading: pocas señales y mal timing → Sharpe bajo.

        - Un μ grande produce β ágil que sigue el cointegración dinámica real.
          → Spread oscila más pero de forma estacionaria (mean-reverting).
          → Autocorrelación negativa → el spread REVIERTE hacia su media.
          → Trading: señales limpias, buena media-reversión → Sharpe alto.

        El baseline VS-NLMS usa μ ≈ 0.43 de media (Kwong-Johnston empuja μ alto
        porque los errores son consistentemente del mismo signo). Este μ alto
        es exactamente lo que produce el buen Sharpe = 0.92.

        Con target = mean_reversion, el modelo aprende a recomendar el μ que
        produce el spread más estacionario → alineado con el objetivo de trading.

    ──────────────────────────────────────────────────────────────────────────
    Features X[t]:
        Se construyen a partir de un filtro de REFERENCIA (μ = mediana de candidatos).
        Las features capturan el estado actual del proceso de filtrado:

        1. |e_ref[t-1]|, ..., |e_ref[t-W]|  → magnitud reciente del error
           (W = feature_window)
        2. std(|e_ref[t-W:t]|)               → volatilidad del error reciente
        3. sign(e_ref[t-1]) · sign(e_ref[t-2]) → correlación de signos (underfitting?)
        4. autocorr_ref (lag-1)               → autocorrelación lag-1 del error de referencia
        5. autocorr_ref (lag-2)               → autocorrelación lag-2 (detecta ciclos de 2 pasos)
        6. vol_ratio                          → std_reciente / std_larga (expansión/contracción)
        7. error_trend                        → pendiente de |e| sobre feature_window (normalizada)
        8. mean_sign                          → media de sign(e) en la ventana reciente (sesgo)

    Total features: feature_window + 7 = 12 (con feature_window=5)

    Parámetros
    ----------
    price_x : np.ndarray
        Precios del activo X (precio_x del pair). Shape (n,).
    price_y : np.ndarray
        Precios del activo Y (precio_y del pair). Shape (n,).
    mu_candidates : list of float
        Lista de μ fijos para la competición de filtros.
    feature_window : int
        Número de errores pasados a incluir como features.
    target_window : int
        Ventana (días) para calcular la autocorrelación del spread en el target.
        20 días ≈ 1 mes de trading. Mayor ventana = target más estable pero menos reactivo.
    eps : float
        Regularización anti-división-por-cero.

    Retorna
    -------
    (model, scaler) : (MLPRegressor, StandardScaler)
        Modelo entrenado y scaler ajustado sobre los datos de TRAIN.
        Listos para pasarse a ML_VSNLMSFilter.
    """
    n = len(price_x)
    n_candidates = len(mu_candidates)

    # Necesitamos al menos target_window pasos para calcular autocorrelación.
    min_start = max(feature_window, target_window)

    # ── Paso 1: correr N filtros NLMS en paralelo ──────────────────────────────
    weights = np.zeros(n_candidates)           # w_i, un escalar por filtro
    errors  = np.zeros((n, n_candidates))      # e_i[t] para todos i y t

    # Filtro de referencia (μ = mediana de candidatos).
    mu_ref      = float(np.median(mu_candidates))
    w_ref       = 0.0
    errors_ref  = np.zeros(n)

    for t in range(n):
        x = float(price_x[t])
        y = float(price_y[t])
        norm = x * x + eps

        e_ref         = y - w_ref * x
        errors_ref[t] = e_ref
        w_ref         = w_ref + mu_ref * e_ref * x / norm

        for i, mu in enumerate(mu_candidates):
            e_i          = y - weights[i] * x
            errors[t, i] = e_i
            weights[i]   = weights[i] + mu * e_i * x / norm

    # ── Paso 2: construir targets usando mean-reversion ────────────────────────
    # Para cada t >= target_window:
    #   Calcular autocorrelación lag-1 del spread de cada filtro_i en [t-W, t).
    #   El ganador es el filtro con autocorrelación MÁS NEGATIVA (más mean-reverting).
    #   Target Y[t] = mu_candidates[ganador].
    def _lag1_autocorr(series: np.ndarray) -> float:
        """Autocorrelación lag-1 de una serie. Retorna 0 si std=0."""
        if len(series) < 4 or np.std(series) < 1e-10:
            return 0.0
        return float(np.corrcoef(series[:-1], series[1:])[0, 1])

    def _lag2_autocorr(series: np.ndarray) -> float:
        """Autocorrelación lag-2 de una serie. Retorna 0 si std=0."""
        if len(series) < 5 or np.std(series) < 1e-10:
            return 0.0
        return float(np.corrcoef(series[:-2], series[2:])[0, 1])

    # ── Paso 3: construir matriz de features X y vector de targets Y ──────────
    X_features = []
    Y_targets  = []

    for t in range(min_start, n):

        # ── Target: μ del filtro con spread más mean-reverting ─────────────────
        # Calcular autocorrelación lag-1 del spread de cada filtro en [t-W, t).
        # El más negativo = más mean-reverting = ganador.
        acorrs = np.array([
            _lag1_autocorr(errors[t - target_window : t, i])
            for i in range(n_candidates)
        ])
        best_i    = int(np.argmin(acorrs))   # más negativo = más mean-reverting
        target_mu = float(mu_candidates[best_i])

        # ── Features del filtro de referencia ─────────────────────────────────
        recent_errors_ref = errors_ref[t - feature_window : t]
        abs_recent        = np.abs(recent_errors_ref)
        recent_long       = errors_ref[t - target_window : t]

        feats = list(abs_recent)

        # Volatilidad del error reciente
        std_recent = float(np.std(abs_recent)) if len(abs_recent) > 1 else 0.0
        feats.append(std_recent)

        # Correlación de signos entre los dos últimos errores
        if len(recent_errors_ref) >= 2:
            sign_corr = float(np.sign(recent_errors_ref[-1]) *
                              np.sign(recent_errors_ref[-2]))
        else:
            sign_corr = 0.0
        feats.append(sign_corr)

        # Autocorrelación lag-1 del filtro de referencia (feature alineada con el target)
        acorr_ref = _lag1_autocorr(recent_long)
        feats.append(acorr_ref)

        # ── Nuevas features (Option 4) ─────────────────────────────────────────

        # Autocorrelación lag-2 (detecta ciclos de periodo 2 en el spread)
        acorr_ref_lag2 = _lag2_autocorr(recent_long)
        feats.append(acorr_ref_lag2)

        # Ratio de volatilidad: std reciente / std largo (régimen de expansión)
        std_long = float(np.std(np.abs(recent_long))) + eps
        feats.append(float(std_recent / std_long))

        # Tendencia del error: pendiente lineal de |e| en la ventana corta (normalizada)
        if len(abs_recent) > 1:
            x_idx = np.arange(len(abs_recent), dtype=float)
            slope = float(np.polyfit(x_idx, abs_recent, 1)[0])
            feats.append(slope / std_long)
        else:
            feats.append(0.0)

        # Sesgo direccional: media de sign(e) en la ventana reciente
        feats.append(float(np.mean(np.sign(recent_errors_ref))))

        X_features.append(feats)
        Y_targets.append(target_mu)

    X_new = np.array(X_features)   # (n - min_start, feature_window + 3)
    Y_new = np.array(Y_targets)    # (n - min_start,)

    # ── Paso 4: combinar con datos acumulados (si los hay) ────────────────────
    # X_extra / Y_extra son muestras de runs anteriores (otros pares).
    # Entrenar sobre el pool completo → el modelo generaliza mejor.
    if X_extra is not None and len(X_extra) > 0:
        X = np.vstack([X_extra, X_new])
        Y = np.concatenate([Y_extra, Y_new])
    else:
        X = X_new
        Y = Y_new

    # ── Paso 5: escalar y entrenar el modelo ──────────────────────────────────
    # Las features de error ya están normalizadas por precio → escala similar
    # a autocorr y sign_corr (ambos en [-1, 1]).
    # StandardScaler añade la normalización final para el MLP.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MLPRegressor: tarea de REGRESIÓN (predecir μ continuo ∈ [μ_min, μ_max]).
    # Arquitectura (32, 16): dos capas ocultas.
    # - 32 neuronas: captura relaciones no lineales entre las 8 features.
    # - 16 neuronas: comprime la representación antes de la salida.
    # max_iter=1000: suficiente con early_stopping.
    # early_stopping=True: para automáticamente si el val_loss deja de mejorar.
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        n_iter_no_change=20,
    )
    model.fit(X_scaled, Y)

    # Retorna también X_new / Y_new para que el caller pueda acumularlos en disco.
    # X_new / Y_new son SOLO los datos generados en este call (no X_extra).
    return model, scaler, X_new, Y_new


# ── Clase ML_VSNLMSFilter ─────────────────────────────────────────────────────

class ML_VSNLMSFilter:
    """
    Filtro NLMS con step-size μ controlado por Machine Learning.

    Misma interfaz que NLMSFilter, RLSFilter, LeakyNLMSFilter y VSNLMSFilter:
        __init__, update(x, y), run(X, y)

    ¿Qué es diferente respecto a VSNLMSFilter?
    ────────────────────────────────────────────
    VSNLMSFilter usa esta heurística en cada paso:
        corr = sign(e[t]) · sign(e[t-1])
        μ[t] = clip(α·μ[t-1] + γ·corr, μ_min, μ_max)

    ML_VSNLMSFilter reemplaza esa heurística por:
        features = [|e[t-1]|, ..., |e[t-W]|, std(|e|), sign_corr]
        μ[t] = clip(model.predict(features), μ_min, μ_max)

    La ecuación de actualización de pesos NO cambia:
        w[t+1] = w[t] + μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)

    Warm-up: durante los primeros feature_window pasos (no hay suficiente historial),
    el filtro usa mu_fallback (un valor razonable por defecto = mediana de candidatos).

    Parámetros
    ----------
    n_taps : int
        Número de coeficientes (1 para un hedge ratio escalar).
    model : MLPRegressor
        Modelo entrenado por train_mu_predictor(). Si es None, usa mu_fallback siempre.
    scaler : StandardScaler
        Scaler ajustado en train. Si es None, no escala (no recomendado).
    mu_min : float
        Step size mínimo. Clip de seguridad para la predicción del modelo.
    mu_max : float
        Step size máximo. Clip de seguridad para la predicción del modelo.
    feature_window : int
        Número de errores pasados a usar como features (debe coincidir con el
        valor usado en train_mu_predictor).
    mu_fallback : float
        μ usado durante el warm-up (primeros feature_window pasos) y si el
        modelo no está disponible.
    eps : float
        Regularización anti-división-por-cero en la normalización.
    """

    def __init__(
        self,
        n_taps: int = 1,
        model: MLPRegressor = None,
        scaler: StandardScaler = None,
        mu_min: float = MU_MIN_DEFAULT,
        mu_max: float = MU_MAX_DEFAULT,
        feature_window: int = DEFAULT_FEATURE_WINDOW,
        mu_fallback: float = 0.05,
        eps: float = 1e-8,
    ):
        self.n_taps         = n_taps
        self.model          = model
        self.scaler         = scaler
        self.mu_min         = mu_min
        self.mu_max         = mu_max
        self.feature_window = feature_window
        self.mu_fallback    = mu_fallback
        self.eps            = eps

        # Estado del filtro
        self.weights        = np.zeros(n_taps)
        self.mu             = mu_fallback           # μ actual (para mu_history)
        # Guardamos max(feature_window, target_window) + 2 errores para poder
        # calcular la autocorrelación lag-1 sobre target_window pasos (feature nueva).
        self._error_history = []
        self._target_window = DEFAULT_TARGET_WINDOW  # para la feature de autocorr

    def _build_features(self) -> np.ndarray | None:
        """
        Construye el vector de features a partir del historial de errores.

        Mismas features que en train_mu_predictor() para coherencia
        entre entrenamiento e inferencia.

        Features:
            1..W:  |e[t-1]|, ..., |e[t-W]|      (magnitud reciente)
            W+1:   std(|e[t-W:t]|)               (volatilidad reciente)
            W+2:   sign(e[t-1])·sign(e[t-2])     (correlación de signos)
            W+3:   autocorr lag-1(e[t-T:t])       (mean-reversion del spread)
            W+4:   autocorr lag-2(e[t-T:t])       (ciclos de periodo 2)
            W+5:   std_recent / std_long           (régimen de volatilidad)
            W+6:   slope(|e[t-W:t]|) / std_long   (tendencia del error, normalizada)
            W+7:   mean(sign(e[t-W:t]))            (sesgo direccional)

        Total: feature_window + 7 = 12 (con feature_window=5)

        Retorna None si no hay suficiente historial (warm-up).
        """
        min_needed = max(self.feature_window, self._target_window)
        if len(self._error_history) < min_needed:
            return None

        recent_w   = np.array(self._error_history[-self.feature_window:], dtype=float)
        abs_recent = np.abs(recent_w)
        recent_t   = np.array(self._error_history[-self._target_window:], dtype=float)

        feats = list(abs_recent)

        # Volatilidad reciente
        std_recent = float(np.std(abs_recent)) if len(abs_recent) > 1 else 0.0
        feats.append(std_recent)

        # Correlación de signos
        if len(recent_w) >= 2:
            sign_corr = float(np.sign(recent_w[-1]) * np.sign(recent_w[-2]))
        else:
            sign_corr = 0.0
        feats.append(sign_corr)

        # Autocorrelación lag-1 sobre target_window
        if len(recent_t) >= 4 and np.std(recent_t) > 1e-10:
            acorr1 = float(np.corrcoef(recent_t[:-1], recent_t[1:])[0, 1])
        else:
            acorr1 = 0.0
        feats.append(acorr1)

        # Autocorrelación lag-2 sobre target_window
        if len(recent_t) >= 5 and np.std(recent_t) > 1e-10:
            acorr2 = float(np.corrcoef(recent_t[:-2], recent_t[2:])[0, 1])
        else:
            acorr2 = 0.0
        feats.append(acorr2)

        # Ratio de volatilidad: std_reciente / std_largo
        std_long = float(np.std(np.abs(recent_t))) + self.eps
        feats.append(float(std_recent / std_long))

        # Tendencia del error (pendiente normalizada)
        if len(abs_recent) > 1:
            x_idx = np.arange(len(abs_recent), dtype=float)
            slope = float(np.polyfit(x_idx, abs_recent, 1)[0])
            feats.append(slope / std_long)
        else:
            feats.append(0.0)

        # Sesgo direccional: media de sign(e) en la ventana reciente
        feats.append(float(np.mean(np.sign(recent_w))))

        return np.array(feats)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        """
        Un paso del filtro adaptativo con μ predicho por el modelo.

        Flujo:
            1. Predicción: ŷ[t] = w[t]ᵀ · x[t]
            2. Error:      e[t]  = y[t] - ŷ[t]
            3. Guardar e[t] en historial
            4. Si hay suficiente historial → pedir μ[t] al modelo
               Si no                      → usar mu_fallback
            5. Clip μ[t] a [mu_min, mu_max]
            6. Actualizar pesos: w[t+1] = w[t] + μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)

        Parámetros
        ----------
        x : array-like, shape (n_taps,)
            Precio del activo X en el instante t.
        y : float
            Precio del activo Y en el instante t.

        Retorna
        -------
        (y_hat, error) : (float, float)
            Predicción y error de predicción en el instante t.
        """
        x = np.asarray(x, dtype=float)

        # 1. Predicción con pesos actuales
        y_hat = float(np.dot(self.weights, x))

        # 2. Error de predicción
        error = float(y) - y_hat

        # 3. Guardar error y precio en historial.
        #    Mantenemos max(feature_window, target_window) + 1 entradas.
        max_history = max(self.feature_window, self._target_window) + 1
        self._error_history.append(error)
        if len(self._error_history) > max_history:
            self._error_history.pop(0)

        # 4. Determinar μ[t]
        if self.model is not None and self.scaler is not None:
            features = self._build_features()

            if features is not None:
                # El modelo predice μ → escalar con el mismo scaler del train
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                mu_pred = float(self.model.predict(features_scaled)[0])
                # Clip de seguridad: el modelo puede extrapolar fuera del rango razonable
                self.mu = float(np.clip(mu_pred, self.mu_min, self.mu_max))
            else:
                # Warm-up: no hay suficiente historial → usar fallback
                self.mu = self.mu_fallback
        else:
            # Sin modelo → comportarse como NLMS estándar con mu_fallback
            self.mu = self.mu_fallback

        # 5. Actualizar pesos (ecuación NLMS idéntica a todos los demás filtros)
        norm_factor  = float(np.dot(x, x)) + self.eps
        self.weights = self.weights + self.mu * error * x / norm_factor

        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Ejecuta el filtro sobre toda una serie temporal.

        Misma interfaz que run() en NLMSFilter, RLSFilter, etc.

        Parámetros
        ----------
        X : np.ndarray, shape (n, n_taps)
            Precios del activo X.
        y : np.ndarray, shape (n,)
            Precios del activo Y.

        Retorna
        -------
        dict con:
            y_hat           : predicciones
            errors          : errores = spread adaptativo
            weights_history : hedge ratio β en cada paso
            mu_history      : μ predicho por el modelo en cada paso
        """
        n               = len(y)
        y_hat_arr       = np.zeros(n)
        errors_arr      = np.zeros(n)
        weights_history = np.zeros((n, self.n_taps))
        mu_history      = np.zeros(n)

        for t in range(n):
            y_hat_arr[t], errors_arr[t] = self.update(X[t], y[t])
            weights_history[t] = self.weights.copy()
            mu_history[t]      = self.mu

        return {
            "y_hat":           y_hat_arr,
            "errors":          errors_arr,
            "weights_history": weights_history,
            "mu_history":      mu_history,
        }
