"""
scan_universe.py — Scanner de universo de pares para la estrategia NLMS.

Para cada par del universo:
    1. Descarga datos (yfinance, 2010-2026)
    2. Corre el walk-forward completo con los tres filtros:
       - Johansen cointegración al 95%
       - Half-life [3, 60] días
       - Train Sharpe >= 0.30
    3. Reporta tabla resumen: folds operados, Sharpe ML, drawdown, operable ahora

Uso:
    python scan_universe.py                  # universo completo
    python scan_universe.py --sector energy  # solo un sector
    python scan_universe.py --min-folds 2    # mínimo 2 folds operados
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

# ── Universo de pares por sector ───────────────────────────────────────────────
UNIVERSE = {
    "consumer_staples": [
        ("KO",   "PEP",  "Coca-Cola / Pepsi"),
        ("PG",   "CL",   "P&G / Colgate"),
        ("PG",   "KMB",  "P&G / Kimberly-Clark"),
        ("COST", "WMT",  "Costco / Walmart"),
    ],
    "energy": [
        ("XOM",  "CVX",  "Exxon / Chevron"),
        ("COP",  "MRO",  "ConocoPhillips / Marathon"),
        ("SLB",  "HAL",  "Schlumberger / Halliburton"),
        ("BP",   "SHEL", "BP / Shell"),
    ],
    "financials": [
        ("JPM",  "BAC",  "JPMorgan / BofA"),
        ("GS",   "MS",   "Goldman / Morgan Stanley"),
        ("BLK",  "SCHW", "BlackRock / Schwab"),
        ("CB",   "TRV",  "Chubb / Travelers"),
    ],
    "technology": [
        ("MSFT", "ORCL", "Microsoft / Oracle"),
        ("AMD",  "INTC", "AMD / Intel"),
        ("QCOM", "AVGO", "Qualcomm / Broadcom"),
        ("IBM",  "ORCL", "IBM / Oracle"),
    ],
    "healthcare": [
        ("JNJ",  "ABT",  "J&J / Abbott"),
        ("PFE",  "MRK",  "Pfizer / Merck"),
        ("UNH",  "CVS",  "UnitedHealth / CVS"),
        ("BMY",  "ABBV", "BMS / AbbVie"),
    ],
    "utilities": [
        ("NEE",  "DUK",  "NextEra / Duke"),
        ("SO",   "D",    "Southern / Dominion"),
        ("AEP",  "EXC",  "AEP / Exelon"),
    ],
    "retail": [
        ("WMT",  "TGT",  "Walmart / Target"),
        ("HD",   "LOW",  "Home Depot / Lowe's"),
        ("NKE",  "UAA",  "Nike / Under Armour"),
    ],
    "telecom": [
        ("T",    "VZ",   "AT&T / Verizon"),
        ("TMUS", "VZ",   "T-Mobile / Verizon"),
    ],
    "industrials": [
        ("CAT",  "DE",   "Caterpillar / Deere"),
        ("HON",  "MMM",  "Honeywell / 3M"),
        ("UPS",  "FDX",  "UPS / FedEx"),
        ("BA",   "LMT",  "Boeing / Lockheed"),
    ],
}


def download_pair(ticker_x: str, ticker_y: str,
                  start: str = "2010-01-01", end: str = "2026-04-01") -> pd.DataFrame | None:
    """Descarga precios ajustados de dos tickers y los alinea."""
    try:
        dx = yf.download(ticker_x, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"].squeeze()
        dy = yf.download(ticker_y, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"].squeeze()
        df = pd.concat([dx.rename("price_x"), dy.rename("price_y")], axis=1).dropna()
        if len(df) < 1200:   # mínimo ~5 años de datos
            return None
        df.index.name = "date"
        df = df.reset_index()
        df.columns = ["date", "price_x", "price_y"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception:
        return None


def run_walkforward(csv_path: str, pair_name: str) -> dict | None:
    """
    Corre el walk-forward para un par y devuelve métricas resumidas.
    Usa subprocess para reutilizar walk_forward_ml_mu.py sin modificarlo.
    """
    cmd = [
        sys.executable, "walk_forward_ml_mu.py",
        f"--data={csv_path}",
        f"--pair={pair_name}",
        "--no-background",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None

    # Parsear métricas del output
    metrics = {"pair": pair_name}

    # Contar folds operados vs skipped
    folds_run    = len([l for l in output.split("\n") if "crit=" in l and "skipped" not in l])
    folds_skip   = len([l for l in output.split("\n") if "skipped" in l])
    folds_total  = folds_run + folds_skip
    metrics["folds_total"]  = folds_total
    metrics["folds_traded"] = folds_run

    if folds_run == 0:
        metrics["tradeable"] = False
        return metrics

    # Extraer métricas de la tabla final
    for line in output.split("\n"):
        line = line.strip()
        if "Sharpe ratio" in line:
            nums = [x for x in line.replace("│","").split() if _is_number(x)]
            if len(nums) >= 2:
                metrics["sharpe_baseline"] = float(nums[0])
                metrics["sharpe_ml"]       = float(nums[1])
        elif "Total OOS return" in line:
            nums = [x.rstrip("%") for x in line.replace("│","").split() if x.rstrip("%").lstrip("-").replace(".","").isdigit() or (x.rstrip("%").startswith("-") and x.rstrip("%")[1:].replace(".","").isdigit())]
            vals = []
            for x in line.replace("│","").split():
                x = x.strip().rstrip("%")
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
            if len(vals) >= 2:
                metrics["return_baseline"] = vals[0]
                metrics["return_ml"]       = vals[1]
        elif "Max drawdown" in line:
            vals = []
            for x in line.replace("│","").split():
                x = x.strip().rstrip("%")
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
            if len(vals) >= 2:
                metrics["dd_baseline"] = vals[0]
                metrics["dd_ml"]       = vals[1]

    metrics["tradeable"] = metrics.get("sharpe_ml", -99) > 0.5
    return metrics


def _is_number(s: str) -> bool:
    try:
        float(s.replace("%", "").replace("│", ""))
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector",    default=None, help="Sector a escanear (ej: energy)")
    parser.add_argument("--min-folds", default=2, type=int, help="Mínimo folds operados para reportar")
    args = parser.parse_args()

    sectors = {args.sector: UNIVERSE[args.sector]} if args.sector and args.sector in UNIVERSE else UNIVERSE

    data_dir = Path("data/scan")
    data_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_pairs = sum(len(v) for v in sectors.values())
    done = 0

    print(f"\n{'='*70}")
    print(f"  SCANNER DE UNIVERSO — {total_pairs} pares en {len(sectors)} sectores")
    print(f"{'='*70}\n")

    for sector, pairs in sectors.items():
        print(f"── {sector.upper()} ──")
        for tx, ty, desc in pairs:
            done += 1
            print(f"  [{done:2d}/{total_pairs}] {desc:<30}", end=" ", flush=True)

            # Descargar datos
            fname = f"{tx.lower()}_{ty.lower()}.csv"
            fpath = data_dir / fname
            if not fpath.exists():
                df = download_pair(tx, ty)
                if df is None:
                    print("✗ sin datos")
                    continue
                df.to_csv(fpath, index=False)

            # Walk-forward
            metrics = run_walkforward(str(fpath), desc)
            if metrics is None:
                print("✗ timeout")
                continue

            ft = metrics.get("folds_traded", 0)
            if ft == 0:
                print(f"✗ 0/{metrics.get('folds_total',0)} folds pasan filtros")
                continue

            sharpe_ml = metrics.get("sharpe_ml", float("nan"))
            ret_ml    = metrics.get("return_ml", float("nan"))
            dd_ml     = metrics.get("dd_ml", float("nan"))
            tag = "✓ OPERABLE" if metrics.get("tradeable") else "~ marginal"
            print(f"{tag}  folds={ft}/{metrics['folds_total']}  "
                  f"Sharpe={sharpe_ml:+.2f}  Return={ret_ml:+.0f}%  DD={dd_ml:.0f}%")
            metrics["sector"] = sector
            results.append(metrics)
        print()

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    if not results:
        print("Ningún par pasó los filtros.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res[df_res["folds_traded"] >= args.min_folds].copy()
    df_res = df_res.sort_values("sharpe_ml", ascending=False)

    print(f"\n{'='*70}")
    print(f"  RESUMEN — pares con ≥{args.min_folds} folds operados")
    print(f"{'='*70}")
    print(f"\n{'Par':<32} {'Sector':<20} {'Folds':>6} {'Sharpe ML':>10} {'Return ML':>10} {'DD ML':>8}")
    print("-"*90)
    for _, row in df_res.iterrows():
        tag = "✓" if row.get("tradeable") else "~"
        print(f"{tag} {row['pair']:<30} {row['sector']:<20} "
              f"{int(row['folds_traded']):>3}/{int(row['folds_total']):<3} "
              f"{row.get('sharpe_ml', float('nan')):>+9.2f}  "
              f"{row.get('return_ml', float('nan')):>+8.0f}%  "
              f"{row.get('dd_ml', float('nan')):>7.0f}%")

    # Guardar resultados
    out = Path("results/scan_universe.csv")
    out.parent.mkdir(exist_ok=True)
    df_res.to_csv(out, index=False)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()
