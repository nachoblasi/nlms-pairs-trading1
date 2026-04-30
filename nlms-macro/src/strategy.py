"""
strategy.py — Re-exports para compatibilidad con scripts existentes.

Las funciones han sido movidas a módulos más específicos:
    src/cointegration.py  →  johansen_cointegration, compute_halflife
    src/signals.py        →  compute_zscore, generate_signals
    src/backtest.py       →  backtest, performance_metrics
"""

from src.cointegration import johansen_cointegration, compute_halflife
from src.signals import compute_zscore, generate_signals
from src.backtest import backtest, performance_metrics

__all__ = [
    "johansen_cointegration",
    "compute_halflife",
    "compute_zscore",
    "generate_signals",
    "backtest",
    "performance_metrics",
]
