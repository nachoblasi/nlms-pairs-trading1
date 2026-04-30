"""
data_generator.py — Generador de datos sintéticos cointegrados.

Este script crea dos series de precios artificiales (Asset X y Asset Y)
que están cointegradas. La relación es:

    Y(t) = β · X(t) + spread(t)

donde:
    - X(t) es un random walk (no estacionario)
    - spread(t) es un proceso Ornstein-Uhlenbeck (estacionario, mean-reverting)
    - β es el hedge ratio verdadero (constante conocida en datos sintéticos)

La gracia de generar datos sintéticos es que conocemos los parámetros reales
(β, θ, etc.) y podemos evaluar si el filtro NLMS los estima correctamente.
Con datos reales, nunca sabes cuál es el β "verdadero".

Para usar datos reales, reemplaza este módulo por descargas con yfinance.
"""

import numpy as np
import pandas as pd


def generate_cointegrated_pair(
    n_samples: int = 1000,
    beta_true: float = 1.3,
    mean_spread: float = 0.0,
    spread_std: float = 0.5,
    ou_theta: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera dos series de precios cointegradas.

    Parámetros
    ----------
    n_samples : int
        Número de observaciones (días de trading).
    beta_true : float
        Coeficiente de cointegración real. El NLMS intentará estimarlo.
    mean_spread : float
        Media de largo plazo del spread O-U. En 0.0 el spread oscila
        simétricamente alrededor de cero.
    spread_std : float
        Volatilidad de las innovaciones del spread. Más alto = spread más ruidoso.
    ou_theta : float
        Velocidad de reversión a la media del proceso O-U.
        theta=0.05 significa que el 5% de la desviación se corrige cada día.
        theta→0: casi un random walk (peligroso, pierde estacionariedad).
        theta→1: revierte casi instantáneamente (spread muy pegado a la media).
    seed : int
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    pd.DataFrame con columnas: date, price_x, price_y, spread_true
    """

    # ── Generador de números aleatorios con semilla fija ──
    # default_rng es la API moderna de numpy (reemplaza np.random.seed).
    # La semilla garantiza que siempre se generan los mismos datos.
    rng = np.random.default_rng(seed)

    # ── Asset X: random walk geométrico (no estacionario) ──
    # Generamos log-retornos diarios:
    #   media  = 0.0003 (0.03% diario, drift positivo leve — simula que las
    #            acciones suben a largo plazo)
    #   std    = 0.015  (1.5% diario, volatilidad típica de una acción)
    #   tamaño = n_samples (un retorno por cada día)
    returns_x = rng.normal(0.0003, 0.015, n_samples)

    # Convertimos log-retornos a precios:
    #   1. cumsum(returns_x) = suma acumulada de log-retornos
    #      Esto equivale a log(precio_t / precio_0) en cada paso
    #   2. exp(...) convierte de log-espacio a precios reales
    #   3. Multiplicamos por 100 → precio inicial ≈ 100
    # Resultado: movimiento browniano geométrico (GBM),
    # el modelo estándar de precios de activos financieros.
    price_x = 100 * np.exp(np.cumsum(returns_x))

    # ── Spread: proceso Ornstein-Uhlenbeck (estacionario) ──
    # El O-U es el modelo canónico de mean-reversion en tiempo continuo.
    # En su discretización (Euler-Maruyama):
    #
    #   spread[t] = spread[t-1] + θ·(μ - spread[t-1]) + σ·ε[t]
    #
    # Tiene tres componentes:
    #   1. spread[t-1]                 → memoria: donde estaba ayer
    #   2. θ·(μ - spread[t-1])        → fuerza de reversión: empuja hacia μ
    #      - Si spread > μ → término negativo → empuja abajo
    #      - Si spread < μ → término positivo → empuja arriba
    #   3. σ·ε[t]                      → innovación aleatoria (ruido gaussiano)
    #
    # El proceso resultante es estacionario: tiene media y varianza constantes
    # en el tiempo, que es exactamente lo que necesitamos para el NLMS.
    spread = np.zeros(n_samples)  # inicializa en cero
    for t in range(1, n_samples):  # arranca en t=1 porque t=0 ya es 0
        spread[t] = (
            spread[t - 1]                                  # donde estaba ayer
            + ou_theta * (mean_spread - spread[t - 1])     # fuerza de reversión
            + spread_std * rng.normal()                    # innovación aleatoria
        )

    # ── Asset Y: cointegrado con X ──
    # Y = β·X + spread.
    # X es I(1) (integrado de orden 1, no estacionario).
    # spread es I(0) (estacionario).
    # Por tanto Y también es I(1).
    # Pero la combinación lineal Y - β·X = spread es I(0).
    # Eso es exactamente la definición de cointegración:
    #   dos series I(1) cuya combinación lineal es I(0).
    price_y = beta_true * price_x + spread

    # ── Fechas de trading ──
    # bdate_range genera solo días hábiles (lun-vie), excluyendo fines
    # de semana. Es más realista que pd.date_range() que incluiría sábados
    # y domingos en los que no hay mercado.
    dates = pd.bdate_range(start="2020-01-02", periods=n_samples)

    # ── Empaquetamos en DataFrame ──
    df = pd.DataFrame(
        {
            "date": dates,
            "price_x": price_x,
            "price_y": price_y,
            "spread_true": spread,  # guardamos el spread real para comparar con NLMS
        }
    )
    return df


# ── Ejecución directa ──
# Si ejecutas `python src/data_generator.py` directamente, genera y guarda datos.
# Si lo importas desde otro script (como main.py), este bloque NO se ejecuta.
if __name__ == "__main__":
    df = generate_cointegrated_pair()
    df.to_csv("data/synthetic_pair.csv", index=False)
    print(f"Generated {len(df)} samples. Saved to data/synthetic_pair.csv")
    print(f"\nSample:\n{df.head(10)}")
    print(f"\nSpread stats:\n{df['spread_true'].describe()}")
