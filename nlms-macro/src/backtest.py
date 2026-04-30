"""
backtest.py — Motor de backtest y cálculo de métricas de rendimiento.

Funciones:
    backtest()             — calcula retornos diarios de la estrategia
    performance_metrics()  — Sharpe, drawdown, win rate, etc.
"""

import numpy as np
import pandas as pd


def backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    hedge_ratios: np.ndarray,
    target_vol: float = None,
    vol_lookback: int = 21,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Backtest de la estrategia de pairs trading.

    La posición del spread es dollar-neutral:
        Long  spread (+1) = comprar 1 unidad de Y, vender β unidades de X
        Short spread (-1) = vender 1 unidad de Y, comprar β unidades de X

    PnL diario = signal[t-1] · (ret_Y[t] - β[t-1] · ret_X[t])

    Nota: se usa la señal del día anterior (t-1) para evitar lookahead bias.

    Parámetros
    ----------
    df           : pd.DataFrame — columnas: date, price_x, price_y
    signals      : np.ndarray   — señales de generate_signals()
    hedge_ratios : np.ndarray   — β[t] estimados por el filtro NLMS
    target_vol   : float        — vol anualizada objetivo (None = sin escalar)
    vol_lookback : int          — ventana para vol rolling (días)
    max_leverage : float        — cap al multiplicador de volatility targeting

    Retorna
    -------
    pd.DataFrame con columnas:
        date, signal, hedge_ratio, spread_return, strategy_return, cumulative_return
    """
    ret_y = df["price_y"].pct_change().values
    ret_x = df["price_x"].pct_change().values

    spread_returns = ret_y - hedge_ratios * ret_x

    if target_vol is not None:
        target_daily_vol = target_vol / np.sqrt(252)
        rolling_vol = (
            pd.Series(spread_returns)
            .rolling(vol_lookback, min_periods=vol_lookback)
            .std()
            .clip(lower=1e-8)
            .fillna(0)
            .values
        )
        scale            = np.where(rolling_vol > 0, target_daily_vol / rolling_vol, 0.0)
        scale            = np.clip(scale, 0.0, max_leverage)
        effective_signals = signals * scale
    else:
        effective_signals = signals

    strategy_returns     = np.zeros(len(signals))
    strategy_returns[1:] = effective_signals[:-1] * spread_returns[1:]

    cumulative = np.cumprod(1 + strategy_returns) - 1

    return pd.DataFrame({
        "date":               df["date"].values,
        "signal":             signals,
        "hedge_ratio":        hedge_ratios,
        "spread_return":      spread_returns,
        "strategy_return":    strategy_returns,
        "cumulative_return":  cumulative,
    })


def performance_metrics(results: pd.DataFrame) -> dict:
    """
    Calcula métricas de rendimiento sobre el output de backtest().

    Retorna
    -------
    dict con claves (strings formateados):
        total_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, win_rate, n_trades
    """
    returns = results["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    total_return = (1 + returns).prod() - 1
    ann_return   = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol      = np.std(returns) * np.sqrt(252)
    sharpe       = ann_return / ann_vol if ann_vol > 0 else 0

    cum      = np.cumprod(1 + returns)
    peak     = np.maximum.accumulate(cum)
    max_dd   = float(np.min((cum - peak) / peak))

    trades   = returns[returns != 0]
    win_rate = float(np.sum(trades > 0) / len(trades)) if len(trades) > 0 else 0
    n_trades = int(np.sum(np.diff(results["signal"].values) != 0))

    return {
        "total_return":           f"{total_return:.2%}",
        "annualized_return":      f"{ann_return:.2%}",
        "annualized_volatility":  f"{ann_vol:.2%}",
        "sharpe_ratio":           f"{sharpe:.2f}",
        "max_drawdown":           f"{max_dd:.2%}",
        "win_rate":               f"{win_rate:.2%}",
        "n_trades":               n_trades,
    }
