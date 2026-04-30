"""
plots.py — Visualizaciones del proyecto.

Genera una figura con 4 paneles que resumen toda la estrategia:
    Panel 1: Precios del par + hedge ratio adaptativo
    Panel 2: Spread real vs spread estimado por NLMS
    Panel 3: Z-score con umbrales y zonas de señal
    Panel 4: Retorno acumulado de la estrategia

No hay lógica de negocio aquí — solo presentación de resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_all(df: pd.DataFrame, nlms_result: dict, zscore: np.ndarray,
             signals: np.ndarray, backtest_results: pd.DataFrame,
             save_path: str = "results/strategy_report.png"):
    """
    Genera el reporte visual completo de 4 paneles.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos del par (price_x, price_y, spread_true).
    nlms_result : dict
        Output de NLMSFilter.run() — contiene errors y weights_history.
    zscore : np.ndarray
        Z-score del spread adaptativo.
    signals : np.ndarray
        Señales de trading (+1, -1, 0).
    backtest_results : pd.DataFrame
        Output de backtest() — contiene cumulative_return.
    save_path : str
        Ruta donde guardar la imagen.
    """
    # ── Configuración de la figura ──
    # figsize=(14, 16): 14 pulgadas de ancho, 16 de alto (formato vertical).
    fig = plt.figure(figsize=(14, 16))
    # GridSpec divide la figura en una cuadrícula de 4 filas × 1 columna.
    # hspace=0.35 es el espacio vertical entre paneles (35% del alto de un panel).
    gs = gridspec.GridSpec(4, 1, hspace=0.35)

    # Extraemos las fechas para el eje X de todos los paneles.
    dates = df["date"].values

    # ══════════════════════════════════════════════════════════
    # Panel 1: Precios del par + hedge ratio adaptativo
    # ══════════════════════════════════════════════════════════
    # Este panel muestra los dos activos y cómo el NLMS estima β.
    ax1 = fig.add_subplot(gs[0])
    # twinx() crea un SEGUNDO eje Y (a la derecha) que comparte el eje X.
    # Lo necesitamos porque los precios (~100) y el hedge ratio (~1.3)
    # tienen escalas muy diferentes. Sin esto, el hedge ratio sería
    # una línea plana pegada al cero.
    ax1b = ax1.twinx()

    # Precios en el eje Y izquierdo
    ax1.plot(dates, df["price_x"], label="Asset X", alpha=0.8, linewidth=1)
    ax1.plot(dates, df["price_y"], label="Asset Y", alpha=0.8, linewidth=1)
    # Hedge ratio en el eje Y derecho (línea discontinua roja)
    # weights_history[:, 0] = primera columna = β a lo largo del tiempo
    ax1b.plot(dates, nlms_result["weights_history"][:, 0],
              color="tab:red", alpha=0.7, linewidth=1.2, linestyle="--",
              label="NLMS hedge ratio")

    ax1.set_title("Cointegrated Pair & Adaptive Hedge Ratio", fontweight="bold")
    ax1.set_ylabel("Price")
    ax1b.set_ylabel("Hedge Ratio (β)", color="tab:red")
    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")
    ax1.grid(alpha=0.3)  # cuadrícula sutil (30% opacidad)

    # ══════════════════════════════════════════════════════════
    # Panel 2: Spread real vs spread NLMS
    # ══════════════════════════════════════════════════════════
    # Este panel permite evaluar visualmente qué tan bien el NLMS
    # estima el spread. Solo es posible con datos sintéticos
    # (porque conocemos el spread "real").
    ax2 = fig.add_subplot(gs[1])

    # Solo ploteamos el spread real si existe (en datos reales no lo tenemos)
    if "spread_true" in df.columns:
        ax2.plot(dates, df["spread_true"], label="True spread (O-U)",
                 alpha=0.5, linewidth=1)
    # El spread NLMS = errores del filtro = y - β·x
    ax2.plot(dates, nlms_result["errors"], label="NLMS spread (error)",
             alpha=0.8, linewidth=1)

    # Línea horizontal en y=0 como referencia (media del spread)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_title("Spread: True vs NLMS Estimated", fontweight="bold")
    ax2.set_ylabel("Spread")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ══════════════════════════════════════════════════════════
    # Panel 3: Z-score con umbrales y señales
    # ══════════════════════════════════════════════════════════
    # Este es el panel más informativo para entender las señales.
    ax3 = fig.add_subplot(gs[2])

    # Línea del z-score
    ax3.plot(dates, zscore, label="Z-score", color="tab:blue",
             alpha=0.8, linewidth=1)

    # Umbrales de entrada (±2σ) — líneas rojas discontinuas
    ax3.axhline(2.0, color="tab:red", linestyle="--", alpha=0.5, label="Entry (±2σ)")
    ax3.axhline(-2.0, color="tab:red", linestyle="--", alpha=0.5)

    # Umbrales de salida (±0.5σ) — líneas verdes punteadas
    ax3.axhline(0.5, color="tab:green", linestyle=":", alpha=0.5, label="Exit (±0.5σ)")
    ax3.axhline(-0.5, color="tab:green", linestyle=":", alpha=0.5)

    # ── Sombreado de posiciones ──
    # Creamos máscaras booleanas para saber cuándo estamos long/short
    long_mask = signals == 1    # True en días con posición larga
    short_mask = signals == -1  # True en días con posición corta

    # fill_between colorea verticalmente donde la máscara es True.
    # Zona verde = estamos long en el spread.
    # Zona roja = estamos short en el spread.
    ax3.fill_between(dates, ax3.get_ylim()[0], ax3.get_ylim()[1],
                     where=long_mask, alpha=0.1, color="green", label="Long spread")
    ax3.fill_between(dates, ax3.get_ylim()[0], ax3.get_ylim()[1],
                     where=short_mask, alpha=0.1, color="red", label="Short spread")

    ax3.set_title("Z-Score & Trading Signals", fontweight="bold")
    ax3.set_ylabel("Z-Score")
    # ncol=3 pone la leyenda en 3 columnas para que no ocupe mucho espacio
    ax3.legend(loc="upper left", fontsize=8, ncol=3)
    ax3.grid(alpha=0.3)

    # ══════════════════════════════════════════════════════════
    # Panel 4: Retorno acumulado
    # ══════════════════════════════════════════════════════════
    # Muestra la curva de equity (cómo crece/decrece tu dinero).
    ax4 = fig.add_subplot(gs[3])

    ax4.plot(dates, backtest_results["cumulative_return"].values,
             color="tab:purple", linewidth=1.5, label="Strategy")
    # fill_between crea un área sombreada debajo de la curva.
    # Si la curva está por encima de 0, es púrpura claro (ganando).
    # Si baja de 0, muestra visualmente las pérdidas.
    ax4.fill_between(dates, 0, backtest_results["cumulative_return"].values,
                     alpha=0.15, color="tab:purple")

    ax4.axhline(0, color="gray", linestyle=":", alpha=0.5)  # referencia en 0%
    ax4.set_title("Cumulative Strategy Returns", fontweight="bold")
    ax4.set_ylabel("Cumulative Return")
    ax4.set_xlabel("Date")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # ── Formateo de fechas ──
    # Rotamos las etiquetas del eje X 30° para que no se solapen
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis="x", rotation=30)

    # ── Guardar ──
    # dpi=150: resolución (puntos por pulgada). 150 es buena para pantalla.
    # bbox_inches="tight": recorta los márgenes blancos sobrantes.
    # facecolor="white": fondo blanco (matplotlib a veces usa transparente).
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    # close() libera la memoria de la figura. Sin esto, si generas muchas
    # figuras en un bucle, matplotlib acumula memoria.
    plt.close()
    print(f"Report saved to {save_path}")
