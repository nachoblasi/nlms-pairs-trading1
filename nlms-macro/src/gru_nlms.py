"""
gru_nlms.py — Filtro NLMS con μ controlado por GRU (numpy puro, sin PyTorch/TF).

Por qué GRU en lugar de MLP
────────────────────────────
El MLP en ml_nlms.py trata cada timestep de forma independiente:
    μ[t] = MLP(features[t])

El GRU mantiene un estado oculto h[t] que depende de todos los pasos anteriores:
    h[t] = GRU(features[t], h[t-1])
    μ[t] = Wo · h[t] + bo

Esto le permite detectar patrones temporales como:
  - "el spread lleva 5 días con errores del mismo signo → μ alto sostenido"
  - "cambio de régimen hace 10 días → ajustar μ gradualmente"
  El MLP solo ve el estado actual y no puede aprender estos patrones.

Ecuaciones GRU:
    z[t] = σ(Wz·[x[t], h[t-1]] + bz)          # update gate
    r[t] = σ(Wr·[x[t], h[t-1]] + br)          # reset gate
    g[t] = tanh(Wg·[x[t], r[t]⊙h[t-1]] + bg)  # candidato
    h[t] = (1-z[t])⊙h[t-1] + z[t]⊙g[t]        # nuevo hidden state
    μ[t] = Wo·h[t] + bo                         # predicción

Entrenamiento: BPTT completo + Adam.
Inferencia: paso a paso manteniendo h[t] en la clase GRU_VSNLMSFilter.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.ml_nlms import (
    DEFAULT_MU_CANDIDATES,
    DEFAULT_FEATURE_WINDOW,
    DEFAULT_TARGET_WINDOW,
    MU_MIN_DEFAULT,
    MU_MAX_DEFAULT,
)


# ── Activaciones ──────────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

def _sigmoid_d(s):   # derivada de sigmoid dado el valor de salida s
    return s * (1.0 - s)

def _tanh_d(t):      # derivada de tanh dado el valor de salida t
    return 1.0 - t ** 2


# ── Modelo GRU ────────────────────────────────────────────────────────────────

class GRUMuPredictor:
    """
    GRU de una capa para predecir μ.

    Parámetros
    ----------
    input_size  : dimensión del vector de features (12 por defecto)
    hidden_size : neuronas en el estado oculto (32)
    lr          : learning rate de Adam
    n_epochs    : épocas de entrenamiento sobre la secuencia completa de train
    seed        : semilla para reproducibilidad
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        lr: float = 5e-4,
        n_epochs: int = 80,
        seed: int = 42,
    ):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.lr          = lr
        self.n_epochs    = n_epochs

        rng = np.random.default_rng(seed)
        s   = np.sqrt(2.0 / (input_size + hidden_size))   # Xavier

        combined = input_size + hidden_size
        self.Wz = rng.normal(0, s, (hidden_size, combined))
        self.Wr = rng.normal(0, s, (hidden_size, combined))
        self.Wg = rng.normal(0, s, (hidden_size, combined))
        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)

        # Capa de salida: init pequeño, bias cerca de la mediana de μ candidatos
        self.Wo = rng.normal(0, 0.01, hidden_size)
        self.bo = np.float64(0.15)

        self._adam_init()

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _adam_init(self):
        shapes = {
            'Wz': self.Wz.shape, 'Wr': self.Wr.shape, 'Wg': self.Wg.shape,
            'bz': self.bz.shape, 'br': self.br.shape, 'bg': self.bg.shape,
            'Wo': self.Wo.shape, 'bo': (),
        }
        self._m = {k: np.zeros(s) for k, s in shapes.items()}
        self._v = {k: np.zeros(s) for k, s in shapes.items()}
        self._t = 0

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self._t += 1
        params = {
            'Wz': self.Wz, 'Wr': self.Wr, 'Wg': self.Wg,
            'bz': self.bz, 'br': self.br, 'bg': self.bg,
            'Wo': self.Wo,
        }
        for k, g in grads.items():
            g = np.clip(g, -clip, clip)
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            delta = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            if k in params:
                params[k] -= delta
            elif k == 'bo':
                self.bo -= float(delta)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray):
        """
        Forward pass completo sobre la secuencia X (T, input_size).
        Devuelve Y_pred y todos los estados cacheados para BPTT.
        """
        T  = len(X)
        H  = np.zeros((T + 1, self.hidden_size))
        Z  = np.zeros((T, self.hidden_size))
        R  = np.zeros((T, self.hidden_size))
        G  = np.zeros((T, self.hidden_size))
        XH = np.zeros((T, self.input_size + self.hidden_size))   # [x, h_prev]
        XRH = np.zeros((T, self.input_size + self.hidden_size))  # [x, r⊙h_prev]
        Y  = np.zeros(T)

        for t in range(T):
            xh       = np.concatenate([X[t], H[t]])
            XH[t]    = xh
            Z[t]     = _sigmoid(self.Wz @ xh + self.bz)
            R[t]     = _sigmoid(self.Wr @ xh + self.br)
            xrh      = np.concatenate([X[t], R[t] * H[t]])
            XRH[t]   = xrh
            G[t]     = np.tanh(self.Wg @ xrh + self.bg)
            H[t + 1] = (1 - Z[t]) * H[t] + Z[t] * G[t]
            Y[t]     = float(self.Wo @ H[t + 1] + self.bo)

        return Y, H, Z, R, G, XH, XRH

    # ── Backward (BPTT) ───────────────────────────────────────────────────────

    def _backward(self, X, Y_pred, Y_target, H, Z, R, G, XH, XRH):
        """BPTT completo. Retorna dict de gradientes."""
        T   = len(X)
        ni  = self.input_size
        dWz = np.zeros_like(self.Wz)
        dWr = np.zeros_like(self.Wr)
        dWg = np.zeros_like(self.Wg)
        dbz = np.zeros_like(self.bz)
        dbr = np.zeros_like(self.br)
        dbg = np.zeros_like(self.bg)
        dWo = np.zeros_like(self.Wo)
        dbo = 0.0
        dH_next = np.zeros(self.hidden_size)

        for t in reversed(range(T)):
            # Gradiente de la pérdida MSE respecto a la salida
            dY = 2.0 * (Y_pred[t] - Y_target[t]) / T

            # Capa de salida lineal
            dWo += dY * H[t + 1]
            dbo += dY
            dH   = self.Wo * dY + dH_next       # gradiente total sobre h[t+1]

            # h[t+1] = (1-z)⊙h[t] + z⊙g
            dG       = dH * Z[t]
            dZ       = dH * (G[t] - H[t])
            dH_prev  = dH * (1 - Z[t])          # path directo h[t]

            # g = tanh(Wg·[x, r⊙h] + bg)
            dg_pre   = dG * _tanh_d(G[t])
            dWg     += np.outer(dg_pre, XRH[t])
            dbg     += dg_pre
            dxrh     = self.Wg.T @ dg_pre
            dR_t     = dxrh[ni:] * H[t]         # a través del reset gate
            dH_prev += dxrh[ni:] * R[t]          # h[t] directo a través de Wg

            # z = sigmoid(Wz·[x, h] + bz)
            dz_pre   = dZ * _sigmoid_d(Z[t])
            dWz     += np.outer(dz_pre, XH[t])
            dbz     += dz_pre
            dH_prev += (self.Wz.T @ dz_pre)[ni:]

            # r = sigmoid(Wr·[x, h] + br)
            dr_pre   = dR_t * _sigmoid_d(R[t])
            dWr     += np.outer(dr_pre, XH[t])
            dbr     += dr_pre
            dH_prev += (self.Wr.T @ dr_pre)[ni:]

            dH_next = dH_prev

        return {
            'Wz': dWz, 'Wr': dWr, 'Wg': dWg,
            'bz': dbz, 'br': dbr, 'bg': dbg,
            'Wo': dWo, 'bo': np.float64(dbo),
        }

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GRUMuPredictor":
        """
        Entrena el GRU sobre la secuencia completa X (T, input_size), Y (T,).
        Guarda el estado de menor pérdida de validación.
        """
        # Separar 10% final como validación (respeta orden temporal)
        val_split = max(1, int(len(X) * 0.1))
        X_tr, Y_tr = X[:-val_split], Y[:-val_split]
        X_va, Y_va = X[-val_split:], Y[-val_split:]

        best_val_loss = np.inf
        best_state    = self._save_state()

        for epoch in range(self.n_epochs):
            # Forward + backward sobre train
            Y_pred, H, Z, R, G, XH, XRH = self._forward(X_tr)
            grads = self._backward(X_tr, Y_pred, Y_tr, H, Z, R, G, XH, XRH)
            self._adam_step(grads)

            # Validación (sin gradiente)
            Y_va_pred, *_ = self._forward(X_va)
            val_loss = float(np.mean((Y_va_pred - Y_va) ** 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = self._save_state()

        self._load_state(best_state)
        return self

    def _save_state(self):
        return {k: getattr(self, k).copy() if hasattr(getattr(self, k), 'copy')
                else float(getattr(self, k))
                for k in ('Wz','Wr','Wg','bz','br','bg','Wo','bo')}

    def _load_state(self, state):
        for k, v in state.items():
            setattr(self, k, np.array(v) if k != 'bo' else np.float64(v))

    # ── Inferencia paso a paso ────────────────────────────────────────────────

    def step(self, x: np.ndarray, h: np.ndarray):
        """
        Un paso del GRU para inferencia online.
        x : (input_size,) — features actuales
        h : (hidden_size,) — hidden state anterior
        Retorna: (μ_predicho, h_nuevo)
        """
        xh   = np.concatenate([x, h])
        z    = _sigmoid(self.Wz @ xh + self.bz)
        r    = _sigmoid(self.Wr @ xh + self.br)
        xrh  = np.concatenate([x, r * h])
        g    = np.tanh(self.Wg @ xrh + self.bg)
        h_new = (1 - z) * h + z * g
        mu   = float(self.Wo @ h_new + self.bo)
        return mu, h_new


# ── Función de entrenamiento (misma interfaz que train_mu_predictor) ──────────

def train_gru_predictor(
    price_x: np.ndarray,
    price_y: np.ndarray,
    mu_candidates: list = DEFAULT_MU_CANDIDATES,
    feature_window: int = DEFAULT_FEATURE_WINDOW,
    target_window: int  = DEFAULT_TARGET_WINDOW,
    hidden_size: int    = 32,
    n_epochs: int       = 80,
    eps: float          = 1e-8,
) -> tuple:
    """
    Genera features/targets (igual que train_mu_predictor) y entrena el GRU.

    Retorna
    -------
    (gru, scaler) : (GRUMuPredictor, StandardScaler)
        Modelo GRU entrenado y scaler para normalizar features en inferencia.
    """
    n            = len(price_x)
    n_candidates = len(mu_candidates)
    min_start    = max(feature_window, target_window)

    # ── Paso 1: filtros NLMS en paralelo (idéntico a train_mu_predictor) ──────
    weights   = np.zeros(n_candidates)
    errors    = np.zeros((n, n_candidates))
    mu_ref    = float(np.median(mu_candidates))
    w_ref     = 0.0
    errors_ref = np.zeros(n)

    for t in range(n):
        x_t   = float(price_x[t])
        y_t   = float(price_y[t])
        norm  = x_t * x_t + eps
        e_ref = y_t - w_ref * x_t
        errors_ref[t] = e_ref
        w_ref = w_ref + mu_ref * e_ref * x_t / norm
        for i, mu in enumerate(mu_candidates):
            e_i          = y_t - weights[i] * x_t
            errors[t, i] = e_i
            weights[i]   = weights[i] + mu * e_i * x_t / norm

    # ── Paso 2: targets (mean-reversion, idéntico a train_mu_predictor) ───────
    def _lag1_autocorr(s):
        if len(s) < 4 or np.std(s) < 1e-10: return 0.0
        return float(np.corrcoef(s[:-1], s[1:])[0, 1])

    def _lag2_autocorr(s):
        if len(s) < 5 or np.std(s) < 1e-10: return 0.0
        return float(np.corrcoef(s[:-2], s[2:])[0, 1])

    # ── Paso 3: construir X_features, Y_targets ───────────────────────────────
    X_features, Y_targets = [], []

    for t in range(min_start, n):
        acorrs = np.array([_lag1_autocorr(errors[t - target_window:t, i])
                           for i in range(n_candidates)])
        target_mu = float(mu_candidates[int(np.argmin(acorrs))])

        recent_w   = errors_ref[t - feature_window:t]
        abs_recent = np.abs(recent_w)
        recent_long = errors_ref[t - target_window:t]

        feats = list(abs_recent)

        std_recent = float(np.std(abs_recent)) if len(abs_recent) > 1 else 0.0
        feats.append(std_recent)

        sign_corr = (float(np.sign(recent_w[-1]) * np.sign(recent_w[-2]))
                     if len(recent_w) >= 2 else 0.0)
        feats.append(sign_corr)

        acorr_ref = _lag1_autocorr(recent_long)
        feats.append(acorr_ref)

        acorr_ref_lag2 = _lag2_autocorr(recent_long)
        feats.append(acorr_ref_lag2)

        std_long = float(np.std(np.abs(recent_long))) + eps
        feats.append(float(std_recent / std_long))

        if len(abs_recent) > 1:
            slope = float(np.polyfit(np.arange(len(abs_recent), dtype=float),
                                     abs_recent, 1)[0])
            feats.append(slope / std_long)
        else:
            feats.append(0.0)

        feats.append(float(np.mean(np.sign(recent_w))))

        X_features.append(feats)
        Y_targets.append(target_mu)

    X = np.array(X_features)   # (T, 12)
    Y = np.array(Y_targets)    # (T,)

    # ── Paso 4: escalar y entrenar el GRU ─────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gru = GRUMuPredictor(
        input_size  = X_scaled.shape[1],
        hidden_size = hidden_size,
        n_epochs    = n_epochs,
    )
    gru.fit(X_scaled, Y)

    return gru, scaler


# ── Clase GRU_VSNLMSFilter ────────────────────────────────────────────────────

class GRU_VSNLMSFilter:
    """
    Filtro NLMS con step-size μ controlado por GRU.

    Misma interfaz que ML_VSNLMSFilter.
    La diferencia clave: el GRU mantiene h[t] entre pasos → memoria temporal.
    """

    def __init__(
        self,
        n_taps: int = 1,
        gru: GRUMuPredictor = None,
        scaler: StandardScaler = None,
        mu_min: float  = MU_MIN_DEFAULT,
        mu_max: float  = MU_MAX_DEFAULT,
        feature_window: int = DEFAULT_FEATURE_WINDOW,
        mu_fallback: float  = 0.15,
        eps: float = 1e-8,
    ):
        self.n_taps         = n_taps
        self.gru            = gru
        self.scaler         = scaler
        self.mu_min         = mu_min
        self.mu_max         = mu_max
        self.feature_window = feature_window
        self.mu_fallback    = mu_fallback
        self.eps            = eps

        self.weights        = np.zeros(n_taps)
        self.mu             = mu_fallback
        self._error_history = []
        self._target_window = DEFAULT_TARGET_WINDOW
        self._h             = None   # hidden state GRU (inicializado en run)

    def _build_features(self) -> np.ndarray | None:
        """Mismas features que ML_VSNLMSFilter._build_features()."""
        min_needed = max(self.feature_window, self._target_window)
        if len(self._error_history) < min_needed:
            return None

        recent_w   = np.array(self._error_history[-self.feature_window:])
        abs_recent = np.abs(recent_w)
        recent_t   = np.array(self._error_history[-self._target_window:])

        feats = list(abs_recent)

        std_recent = float(np.std(abs_recent)) if len(abs_recent) > 1 else 0.0
        feats.append(std_recent)

        if len(recent_w) >= 2:
            feats.append(float(np.sign(recent_w[-1]) * np.sign(recent_w[-2])))
        else:
            feats.append(0.0)

        for lag in [1, 2]:
            s = recent_t
            if len(s) >= lag + 3 and np.std(s) > 1e-10:
                feats.append(float(np.corrcoef(s[:-lag], s[lag:])[0, 1]))
            else:
                feats.append(0.0)

        std_long = float(np.std(np.abs(recent_t))) + self.eps
        feats.append(float(std_recent / std_long))

        if len(abs_recent) > 1:
            slope = float(np.polyfit(np.arange(len(abs_recent), dtype=float),
                                     abs_recent, 1)[0])
            feats.append(slope / std_long)
        else:
            feats.append(0.0)

        feats.append(float(np.mean(np.sign(recent_w))))

        return np.array(feats)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)

        y_hat = float(np.dot(self.weights, x))
        error = float(y) - y_hat

        max_history = max(self.feature_window, self._target_window) + 1
        self._error_history.append(error)
        if len(self._error_history) > max_history:
            self._error_history.pop(0)

        if self.gru is not None and self.scaler is not None:
            features = self._build_features()
            if features is not None:
                feat_scaled = self.scaler.transform(features.reshape(1, -1))[0]
                mu_pred, self._h = self.gru.step(feat_scaled, self._h)
                self.mu = float(np.clip(mu_pred, self.mu_min, self.mu_max))
            else:
                self.mu = self.mu_fallback
        else:
            self.mu = self.mu_fallback

        norm_factor  = float(np.dot(x, x)) + self.eps
        self.weights = self.weights + self.mu * error * x / norm_factor

        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        n = len(y)
        # Inicializar hidden state a cero al principio de cada secuencia
        if self.gru is not None:
            self._h = np.zeros(self.gru.hidden_size)

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
