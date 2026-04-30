"""
walk_forward_ml_mu.py — Walk-Forward: Baseline VS-NLMS vs ML_VSNLMSFilter.

¿Qué hace este script?
──────────────────────
Ejecuta el mismo walk-forward que walk_forward.py pero comparando dos filtros:

    BASELINE    → VSNLMSFilter  (Kwong-Johnston: μ heurístico)
    ML FILTER   → ML_VSNLMSFilter (MLPRegressor: μ predicho por red neuronal)

En cada fold:
    [TRAIN]
        1. Grid search → mejores params del VS-NLMS (lookback, entry_z, exit_z, mu_init)
        2. Entrenar predictor de μ → train_mu_predictor(price_x, price_y) → (model, scaler)
           (corre N filtros NLMS en paralelo, elige el ganador en cada paso, entrena MLP)

    [TEST]
        3. Baseline: VSNLMSFilter con los mejores params (igual que walk_forward.py)
        4. ML:       ML_VSNLMSFilter(model, scaler) con los mismos lookback/entry_z/exit_z
           El filtro ML usa el MLP para decidir μ en cada paso en lugar de la heurística.

Comparación justa:
    - Los parámetros de estrategia (lookback, entry_z, exit_z) son los mismos para ambos.
    - El grid search y el entrenamiento del MLP usan SOLO datos de train.
    - El test es siempre out-of-sample para ambos filtros.
    - La única diferencia es cómo se adapta μ en cada paso.

Reversibilidad:
    Este script (y src/ml_nlms.py) son completamente independientes del proyecto base.
    Para eliminarlos: borrar walk_forward_ml_mu.py y src/ml_nlms.py.
    No modifica ningún archivo existente.

Requisitos: pip install scikit-learn
Ejecutar:   python walk_forward_ml_mu.py
"""

import sys
import numpy as np
import pandas as pd
import time

from src.ml_nlms import train_mu_predictor, DEFAULT_MU_CANDIDATES, DEFAULT_FEATURE_WINDOW, DEFAULT_TARGET_WINDOW
from src.gru_nlms import train_gru_predictor, GRU_VSNLMSFilter
from src.cointegration import johansen_cointegration, compute_halflife
from src.optimizer import (build_vsnlms, build_ml_filter, run_filter_pipeline,
                            optimize_on_train, PARAM_GRID)
import src.model_store as model_store


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DAYS     = 756   # ~3 años
TEST_DAYS      = 252   # ~1 año
ML_MU_FALLBACK = float(np.median(DEFAULT_MU_CANDIDATES))




def compute_oos_metrics(bt_results: pd.DataFrame) -> dict:
    """Calcula métricas de rendimiento sobre un DataFrame de backtest."""
    returns = bt_results["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return {k: 0 for k in ["total_return", "ann_return", "ann_vol",
                                "sharpe", "max_dd", "win_rate", "n_trades"]}

    total   = (1 + returns).prod() - 1
    ann_ret = (1 + total) ** (252 / len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    cum    = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))

    active   = returns[returns != 0]
    win_rate = float(np.sum(active > 0) / len(active)) if len(active) > 0 else 0
    n_trades = int(np.sum(np.diff(bt_results["signal"].values) != 0))

    return {
        "total_return": total,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── [0] Flags de línea de comandos ────────────────────────────────────────
    if "--reset" in sys.argv:
        print("Borrando modelo acumulado...")
        model_store.reset()
        print("Listo. Vuelve a ejecutar sin --reset para empezar de cero.")
        return

    # --data path/to/file.csv  (por defecto: visa_mastercard.csv)
    # --pair="GS / MS"         (nombre del par para mostrar)
    # --train=N                (días de entrenamiento, por defecto 756)
    # --test=N                 (días de test, por defecto 252)
    # --min-sharpe=X           (umbral mínimo de Sharpe train para acumular el fold, default 0.3)
    # --no-background          (entrena solo con datos del fold actual, sin usar el pool acumulado)
    #                          Usar para el "par principal" (V/MA) → evita leakage temporal.
    #                          Los datos nuevos SÍ se guardan al pool igualmente.
    data_file      = "data/visa_mastercard.csv"
    pair_name      = "V / MA"
    train_days     = TRAIN_DAYS
    test_days      = TEST_DAYS
    min_sharpe     = 0.3
    use_background = "--no-background" not in sys.argv
    use_gru        = "--gru" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--data="):
            data_file = arg.split("=", 1)[1]
        elif arg.startswith("--pair="):
            pair_name = arg.split("=", 1)[1]
        elif arg.startswith("--train="):
            train_days = int(arg.split("=", 1)[1])
        elif arg.startswith("--test="):
            test_days = int(arg.split("=", 1)[1])
        elif arg.startswith("--min-sharpe="):
            min_sharpe = float(arg.split("=", 1)[1])

    ml_label = "GRU_VSNLMSFilter" if use_gru else "ML_VSNLMSFilter"
    print("=" * 70)
    print(f"WALK-FORWARD: BASELINE VS-NLMS  vs  {ml_label} ({pair_name})")
    print("=" * 70)

    # ── [1] Cargar datos ──────────────────────────────────────────────────────
    print("\n[1] Cargando datos...")
    try:
        df = pd.read_csv(data_file, parse_dates=["date"])
        print(f"    {len(df)} filas cargadas  [{data_file}]")
    except FileNotFoundError:
        print(f"    ERROR: No se encuentra {data_file}")
        return


    # ── [1b] Cargar datos acumulados del modelo ───────────────────────────────
    X_acc, Y_acc = model_store.load_accumulated_data()
    print(f"    Modelo acumulado: {model_store.info()}")

    total_days = len(df)
    n_folds    = (total_days - train_days) // test_days

    grid_size = (len(PARAM_GRID["filter_param"]) *
                 len(PARAM_GRID["lookback"]) *
                 len(PARAM_GRID["entry_z"]) *
                 len(PARAM_GRID["exit_z"]))

    # ── [2] Configuración ─────────────────────────────────────────────────────
    print(f"\n[2] Configuración:")
    print(f"    Total días:       {total_days}")
    print(f"    Ventana train:    {train_days} días (~{train_days/252:.1f} años)")
    print(f"    Ventana test:     {test_days}  días (~{test_days/252:.1f} año)")
    print(f"    Número de folds:  {n_folds}")
    print(f"    Grid VS-NLMS:     {grid_size} combos/fold")
    print(f"    μ candidatos ML:  {DEFAULT_MU_CANDIDATES}")
    print(f"    Feature window:   {DEFAULT_FEATURE_WINDOW} días")
    print(f"    Target window:    {DEFAULT_TARGET_WINDOW} días (autocorr mean-reversion)")
    print(f"    μ fallback ML:    {ML_MU_FALLBACK:.4f} (mediana candidatos)")
    print(f"    Umbral acum.:     Train Sharpe ≥ {min_sharpe:.2f} (folds bajo umbral no se guardan)")
    if not use_background:
        print(f"    Modo:             --no-background (entrena sin pool externo → sin leakage temporal)")

    # ── [3] Walk-Forward Loop ─────────────────────────────────────────────────
    print(f"\n[3] Ejecutando walk-forward...\n")

    all_baseline_results = []
    all_ml_results       = []
    fold_summaries       = []

    # Estadísticas del μ predicho por el modelo (para análisis posterior).
    all_mu_ml = []

    # Datos de entrenamiento generados en este run (para acumular al final).
    fold_X_new = []
    fold_Y_new = []

    start_time = time.time()

    for fold in range(n_folds):
        train_start = fold * test_days
        train_end   = train_start + train_days
        test_end    = train_end + test_days

        if test_end > total_days:
            break

        # Extraer ventanas de datos.
        df_train = df.iloc[train_start:train_end].reset_index(drop=True)
        df_full  = df.iloc[train_start:test_end].reset_index(drop=True)

        train_dates = (f"{df.iloc[train_start]['date'].strftime('%Y-%m-%d')} → "
                       f"{df.iloc[train_end-1]['date'].strftime('%Y-%m-%d')}")
        test_dates  = (f"{df.iloc[train_end]['date'].strftime('%Y-%m-%d')} → "
                       f"{df.iloc[test_end-1]['date'].strftime('%Y-%m-%d')}")

        # ── FASE TRAIN ────────────────────────────────────────────────────────

        # 0. Test de cointegración de Johansen sobre la ventana de train.
        #    Si el par no está cointegrado al 95%, saltamos el fold → evitamos
        #    tradear una relación rota que solo generaría ruido.
        joh = johansen_cointegration(
            df_train["price_x"].values,
            df_train["price_y"].values,
            significance=0.05,
        )
        coint_tag = f"trace={joh['trace_stat']:.1f} crit={joh['crit_value']:.1f}"
        if not joh["cointegrated"]:
            print(f"    Fold {fold+1:2d}/{n_folds}  Train: {train_dates}  Test: {test_dates}")
            print(f"           ✗ Sin cointegración al 95% ({coint_tag}) — fold skipped")
            continue

        # 0b. Half-life de mean-reversion del spread (segundo gate).
        #     Si el spread revierte demasiado lento (>60d) o demasiado rápido (<3d)
        #     no es operable con la estrategia de trading diario.
        hl = compute_halflife(
            df_train["price_x"].values,
            df_train["price_y"].values,
            min_hl=3.0,
            max_hl=60.0,
        )
        hl_tag = f"hl={hl['halflife']:.1f}d" if hl["halflife"] != float("inf") else "hl=∞"
        if not hl["in_range"]:
            print(f"    Fold {fold+1:2d}/{n_folds}  Train: {train_dates}  Test: {test_dates}")
            print(f"           ✗ Half-life fuera de rango ({hl_tag}, rango=[3,60]d) — fold skipped")
            continue

        # 1. Grid search sobre el VS-NLMS baseline → mejores params de estrategia.
        #    Los params (lookback, entry_z, exit_z) se usarán también para el ML filter.
        best_params = optimize_on_train(df_train)
        if best_params is None:
            print(f"    Fold {fold+1:2d}: no se encontraron parámetros válidos, skipping.")
            continue

        # 1b. Gate de Sharpe mínimo en train.
        #     Si el grid search no encuentra ninguna config con Sharpe > min_sharpe,
        #     el par no tiene alfa in-sample → no tradeamos el fold.
        if best_params["train_sharpe"] < min_sharpe:
            print(f"    Fold {fold+1:2d}/{n_folds}  Train: {train_dates}  Test: {test_dates}")
            print(f"           ✗ Train Sharpe insuficiente ({best_params['train_sharpe']:+.2f} < {min_sharpe:.2f}) — fold skipped")
            continue

        # 2. Entrenar predictor de μ SOLO con datos del train.
        price_x_train = df_train["price_x"].values
        price_y_train = df_train["price_y"].values

        if use_gru:
            gru_model, gru_scaler = train_gru_predictor(
                price_x       = price_x_train,
                price_y       = price_y_train,
                mu_candidates = DEFAULT_MU_CANDIDATES,
                feature_window= DEFAULT_FEATURE_WINDOW,
                target_window = DEFAULT_TARGET_WINDOW,
            )
            X_fold, Y_fold = np.array([]), np.array([])
        else:
            ml_model, ml_scaler, X_fold, Y_fold = train_mu_predictor(
                price_x        = price_x_train,
                price_y        = price_y_train,
                mu_candidates  = DEFAULT_MU_CANDIDATES,
                feature_window = DEFAULT_FEATURE_WINDOW,
                target_window  = DEFAULT_TARGET_WINDOW,
                X_extra        = X_acc if (use_background and len(X_acc) > 0) else None,
                Y_extra        = Y_acc if (use_background and len(Y_acc) > 0) else None,
            )

        # Solo acumular el fold si el Sharpe de train supera el umbral.
        if best_params["train_sharpe"] >= min_sharpe:
            fold_X_new.append(X_fold)
            fold_Y_new.append(Y_fold)

        # ── FASE TEST ─────────────────────────────────────────────────────────
        # Ambos filtros corren sobre la ventana COMPLETA (train + test)
        # para que el filtro esté "caliente" al llegar al periodo de test.

        # 3. BASELINE: VSNLMSFilter con los mejores params del grid search.
        filt_base = build_vsnlms(best_params["filter_param"])
        pipe_base = run_filter_pipeline(
            df_full, filt_base,
            lookback = best_params["lookback"],
            entry_z  = best_params["entry_z"],
            exit_z   = best_params["exit_z"],
        )
        bt_base_full = pipe_base["bt"]
        bt_base_test = bt_base_full.iloc[train_days:].copy()
        baseline_metrics = compute_oos_metrics(bt_base_test)

        # 4. ML / GRU FILTER con el modelo entrenado.
        if use_gru:
            filt_ml = GRU_VSNLMSFilter(
                n_taps      = 1,
                gru         = gru_model,
                scaler      = gru_scaler,
                mu_fallback = ML_MU_FALLBACK,
            )
        else:
            filt_ml = build_ml_filter(ml_model, ml_scaler, ML_MU_FALLBACK)
        pipe_ml = run_filter_pipeline(
            df_full, filt_ml,
            lookback = best_params["lookback"],
            entry_z  = best_params["entry_z"],
            exit_z   = best_params["exit_z"],
        )
        bt_ml_full = pipe_ml["bt"]
        bt_ml_test = bt_ml_full.iloc[train_days:].copy()
        ml_metrics = compute_oos_metrics(bt_ml_test)

        # Guardar μ predicho en el periodo de test para análisis.
        mu_ml_test = pipe_ml["mu_history"][train_days:]
        all_mu_ml.append(mu_ml_test)

        # Acumular resultados OOS.
        all_baseline_results.append(bt_base_test)
        all_ml_results.append(bt_ml_test)

        # Estadísticas de μ del baseline y ML en el test (para comparar adaptación).
        mu_base_test = pipe_base["mu_history"][train_days:]
        mu_base_mean = float(np.mean(mu_base_test))
        mu_ml_mean   = float(np.mean(mu_ml_test))
        mu_ml_std    = float(np.std(mu_ml_test))

        fold_summaries.append({
            "fold":            fold + 1,
            "train_period":    train_dates,
            "test_period":     test_dates,
            "train_sharpe":    best_params["train_sharpe"],
            "baseline_sharpe": baseline_metrics["sharpe"],
            "ml_sharpe":       ml_metrics["sharpe"],
            "mu_base_mean":    mu_base_mean,
            "mu_ml_mean":      mu_ml_mean,
            "mu_ml_std":       mu_ml_std,
        })

        ml_flag = "↑" if ml_metrics["sharpe"] > baseline_metrics["sharpe"] else "↓"
        print(
            f"    Fold {fold+1:2d}/{n_folds}  "
            f"Train: {train_dates}  Test: {test_dates}\n"
            f"           [{coint_tag}  {hl_tag}]  "
            f"lb={best_params['lookback']:3d} "
            f"ez={best_params['entry_z']:.1f} "
            f"xz={best_params['exit_z']:.2f}  "
            f"Train: {best_params['train_sharpe']:+.2f}  │  "
            f"Baseline: {baseline_metrics['sharpe']:+.2f}  "
            f"ML ({ml_flag}): {ml_metrics['sharpe']:+.2f}  │  "
            f"μ_base={mu_base_mean:.3f}  μ_ml={mu_ml_mean:.3f}±{mu_ml_std:.3f}"
        )

    elapsed = time.time() - start_time
    print(f"\n    Completado en {elapsed:.0f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTADOS AGREGADOS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[4] Resultados agregados OUT-OF-SAMPLE:\n")

    if not all_baseline_results:
        print("    No hay resultados para agregar.")
        return

    combined_baseline = pd.concat(all_baseline_results, ignore_index=True)
    combined_ml       = pd.concat(all_ml_results,       ignore_index=True)

    base_m = compute_oos_metrics(combined_baseline)
    ml_m   = compute_oos_metrics(combined_ml)

    summary_df       = pd.DataFrame(fold_summaries)
    avg_train_sharpe = summary_df["train_sharpe"].mean()
    avg_base_sharpe  = summary_df["baseline_sharpe"].mean()
    avg_ml_sharpe    = summary_df["ml_sharpe"].mean()
    degradation_base = 1 - (avg_base_sharpe / avg_train_sharpe) if avg_train_sharpe != 0 else 0
    degradation_ml   = 1 - (avg_ml_sharpe   / avg_train_sharpe) if avg_train_sharpe != 0 else 0

    # Estadísticas agregadas de μ.
    all_mu_ml_flat   = np.concatenate(all_mu_ml) if all_mu_ml else np.array([])
    avg_mu_ml_global = float(np.mean(all_mu_ml_flat)) if len(all_mu_ml_flat) > 0 else 0

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    w   = 18
    sep = "─" * (w * 2 + 36)

    print(f"    ┌{sep}┐")
    print(f"    │  {'Métrica':<26}  {'BASELINE VS-NLMS':>{w}}  {'ML_VSNLMSFilter':>{w}}  │")
    print(f"    ├{sep}┤")
    print(f"    │  {'Total OOS return':<26}  {base_m['total_return']:>{w}.2%}  {ml_m['total_return']:>{w}.2%}  │")
    print(f"    │  {'Return anualizado':<26}  {base_m['ann_return']:>{w}.2%}  {ml_m['ann_return']:>{w}.2%}  │")
    print(f"    │  {'Volatilidad anualizada':<26}  {base_m['ann_vol']:>{w}.2%}  {ml_m['ann_vol']:>{w}.2%}  │")
    print(f"    │  {'Sharpe ratio':<26}  {base_m['sharpe']:>{w}.2f}  {ml_m['sharpe']:>{w}.2f}  │")
    print(f"    │  {'Max drawdown':<26}  {base_m['max_dd']:>{w}.2%}  {ml_m['max_dd']:>{w}.2%}  │")
    print(f"    │  {'Win rate':<26}  {base_m['win_rate']:>{w}.2%}  {ml_m['win_rate']:>{w}.2%}  │")
    print(f"    │  {'Total trades':<26}  {base_m['n_trades']:>{w}d}  {ml_m['n_trades']:>{w}d}  │")
    print(f"    ├{sep}┤")
    print(f"    │  {'Avg Train Sharpe':<26}  {avg_train_sharpe:>{w}.3f}  {avg_train_sharpe:>{w}.3f}  │")
    print(f"    │  {'Avg Test Sharpe':<26}  {avg_base_sharpe:>{w}.3f}  {avg_ml_sharpe:>{w}.3f}  │")
    print(f"    │  {'Degradación train→test':<26}  {degradation_base:>{w}.0%}  {degradation_ml:>{w}.0%}  │")
    print(f"    ├{sep}┤")
    avg_mu_base_global = summary_df["mu_base_mean"].mean()
    avg_mu_ml_std_mean = summary_df["mu_ml_std"].mean()
    print(f"    │  {'μ medio en test (OOS)':<26}  {avg_mu_base_global:>{w}.4f}  {avg_mu_ml_global:>{w}.4f}  │")
    print(f"    │  {'μ std en test (OOS)':<26}  {'—':>{w}}  {avg_mu_ml_std_mean:>{w}.4f}  │")
    n_better = int((summary_df["ml_sharpe"] > summary_df["baseline_sharpe"]).sum())
    print(f"    │  {'Folds ML > Baseline':<26}  {'—':>{w}}  {f'{n_better}/{len(summary_df)}':>{w}}  │")
    print(f"    └{sep}┘")

    # ── Interpretación automática ──────────────────────────────────────────────
    print(f"\n    Interpretación:")
    sharpe_diff = ml_m["sharpe"] - base_m["sharpe"]
    dd_diff     = abs(ml_m["max_dd"]) - abs(base_m["max_dd"])
    ret_diff    = ml_m["total_return"] - base_m["total_return"]

    if sharpe_diff > 0.1:
        print(f"    ✓ ML_VSNLMSFilter MEJORA el Sharpe en +{sharpe_diff:.2f}")
    elif sharpe_diff > 0:
        print(f"    ~ ML_VSNLMSFilter mejora marginalmente el Sharpe (+{sharpe_diff:.2f})")
    else:
        print(f"    ✗ ML_VSNLMSFilter NO mejora el Sharpe ({sharpe_diff:+.2f})")

    if dd_diff < -0.005:
        print(f"    ✓ ML_VSNLMSFilter REDUCE el drawdown en {abs(dd_diff):.1%}")
    elif dd_diff < 0:
        print(f"    ~ ML_VSNLMSFilter reduce marginalmente el drawdown ({dd_diff:+.1%})")
    else:
        print(f"    ~ El drawdown no mejora con ML ({dd_diff:+.1%})")

    if ret_diff > 0.05:
        print(f"    ✓ ML_VSNLMSFilter genera +{ret_diff:.1%} de retorno adicional (OOS)")
    elif ret_diff > 0:
        print(f"    ~ ML_VSNLMSFilter genera marginalmente más retorno (+{ret_diff:.1%})")
    else:
        print(f"    ~ ML_VSNLMSFilter genera menos retorno ({ret_diff:+.1%})")

    mu_diff = avg_mu_ml_global - avg_mu_base_global
    print(f"\n    Análisis de μ:")
    print(f"    → Baseline μ medio: {avg_mu_base_global:.4f} (adaptado por Kwong-Johnston)")
    print(f"    → ML μ medio:       {avg_mu_ml_global:.4f} (predicho por MLP, std={avg_mu_ml_std_mean:.4f})")
    if abs(mu_diff) < 0.005:
        print(f"    → El modelo aprende un μ similar al heurístico ({mu_diff:+.4f} de diferencia).")
    elif mu_diff > 0:
        print(f"    → El modelo usa μ más ALTO (+{mu_diff:.4f}) → más agresivo en adaptación.")
    else:
        print(f"    → El modelo usa μ más BAJO ({mu_diff:.4f}) → más conservador en adaptación.")

    # ── Guardar resultados ─────────────────────────────────────────────────────
    summary_df.to_csv("results/walk_forward_ml_mu_folds.csv", index=False)
    combined_baseline.to_csv("results/walk_forward_ml_mu_baseline_oos.csv", index=False)
    combined_ml.to_csv("results/walk_forward_ml_mu_oos.csv", index=False)

    print(f"\n    Folds guardados en:   results/walk_forward_ml_mu_folds.csv")
    print(f"    Baseline OOS en:      results/walk_forward_ml_mu_baseline_oos.csv")
    print(f"    ML OOS en:            results/walk_forward_ml_mu_oos.csv")

    # ── Guardar datos de entrenamiento acumulados ─────────────────────────────
    if fold_X_new:
        X_run = np.vstack(fold_X_new)
        Y_run = np.concatenate(fold_Y_new)

        if len(X_acc) > 0 and X_acc.shape[1] == X_run.shape[1]:
            X_save = np.vstack([X_acc, X_run])
            Y_save = np.concatenate([Y_acc, Y_run])
        else:
            X_save, Y_save = X_run, Y_run

        model_store.save_accumulated_data(X_save, Y_save)
        n_prev       = len(Y_acc)
        n_new        = len(Y_run)
        folds_saved  = len(fold_X_new)
        folds_total  = len(fold_summaries)
        print(f"\n    Modelo acumulado actualizado:")
        print(f"      Folds acumulados:   {folds_saved}/{folds_total} (train Sharpe ≥ {min_sharpe:.2f})")
        print(f"      Muestras previas:   {n_prev:,}")
        print(f"      Nuevas (este run):  {n_new:,}")
        print(f"      Total guardado:     {len(Y_save):,}  →  {model_store.info()}")

    print(f"\n✓ Listo!")


if __name__ == "__main__":
    main()
