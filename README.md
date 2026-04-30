# NLMS Pairs Trading Strategy
### Adaptive Hedge-Ratio Estimation with Variable Step-Size NLMS Filters and Machine Learning μ Prediction

---

## Abstract

This project implements a **statistical arbitrage pairs trading system** using adaptive digital signal processing filters to estimate a time-varying hedge ratio. The core innovation is the application of the **Variable Step-Size NLMS (VS-NLMS)** filter — a class of algorithms from adaptive filtering theory — to the problem of dynamic cointegration estimation in financial time series.

The system is validated with a **walk-forward optimization** over 16 years of daily data (2010–2026), employing a cascade of three pre-trade filters (Johansen cointegration, half-life, train Sharpe gate) to ensure only structurally sound pair-periods are traded.

A secondary research thread investigates replacing the heuristic μ adaptation rule with a **neural network trained on a parallel-filter competition** (`ML_VSNLMSFilter`). The ML version outperforms the baseline on pairs with moderate signal, while the baseline dominates on extremely clean pairs.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Three-Gate Cointegration Filter](#2-three-gate-cointegration-filter)
3. [Adaptive Filter Theory](#3-adaptive-filter-theory)
4. [The VS-NLMS Filter in Detail](#4-the-vs-nlms-filter-in-detail)
5. [Trading Strategy](#5-trading-strategy)
6. [Walk-Forward Validation Methodology](#6-walk-forward-validation-methodology)
7. [Universe Scan Results — All Pairs Tested](#7-universe-scan-results--all-pairs-tested)
8. [Best Pairs — Detailed Results](#8-best-pairs--detailed-results)
9. [Macro Universe — FX, Bonds, Commodities and Indices](#9-macro-universe--fx-bonds-commodities-and-indices)
10. [Machine Learning Experiments](#10-machine-learning-experiments)
11. [How to Use the Strategy in Practice](#11-how-to-use-the-strategy-in-practice)
12. [Project Structure](#12-project-structure)
13. [Installation and Usage](#13-installation-and-usage)
14. [Portfolio Construction and Results](#14-portfolio-construction-and-results)
15. [References](#15-references)

---

## 1. Problem Formulation

Pairs trading exploits a fundamental insight: two companies in the same industry share common macro risk factors. When both stocks are driven by the same underlying economics, their prices should move together in the long run. Temporary divergences from this equilibrium represent exploitable mispricings.

Formally, we model the relationship as:

```
price_Y[t] = β[t] · price_X[t] + e[t]
```

where:
- `β[t]` is the **hedge ratio** — how many dollars of X we need to hedge one dollar of Y
- `e[t]` is the **spread** — the residual that should be stationary if the pair is cointegrated
- The `[t]` subscript on β denotes that we allow the ratio to **change over time**

The key challenge is estimating `β[t]` in real time without lookahead bias. This is where adaptive filters come in.

**Why not just use OLS?** Ordinary Least Squares computes a single fixed β over the entire sample. But cointegration relationships drift over time. A static β produces a spread that is only locally stationary, generating false signals when β has shifted.

---

## 2. Three-Gate Cointegration Filter

Before trading any fold, three sequential filters must be passed. If any gate fails, the fold is skipped entirely.

### Gate 1 — Johansen Cointegration Test (95% confidence)

We use the **Johansen (1988) maximum likelihood test** rather than the simpler Engle-Granger two-step procedure. Advantages:

- **Symmetry**: does not require choosing which series is the "dependent" variable
- **Higher statistical power** in finite samples
- Directly estimates the number of cointegrating vectors

The trace statistic tests H₀: r = 0 (no cointegrating vector) versus H₁: r ≥ 1. We reject H₀ at 5% significance if:

```
λ_trace = −T · Σᵢ log(1 − λ̂ᵢ)  >  critical_value(95%)  [= 15.495]
```

**Why Johansen and not Engle-Granger?** The Engle-Granger test requires choosing which series is Y and which is X, and gives different results depending on the choice. Johansen treats both symmetrically and has higher power, which is important when working with 756-day windows.

### Gate 2 — Half-Life Filter [3, 60] days

Even if cointegration is confirmed statistically, the spread may mean-revert too slowly (> 60 days: unprofitable, too much time in drawdown) or too fast (< 3 days: execution costs dominate). We estimate the half-life using the AR(1) coefficient of the spread:

```
Δspread[t] = ρ · spread[t−1] + ε[t]
half_life   = −log(2) / log(1 + ρ)
```

Only spreads with `3 ≤ half_life ≤ 60` days are traded.

### Gate 3 — Train Sharpe Gate (≥ 0.30)

Even with valid cointegration and good half-life, the training period Sharpe must be ≥ 0.30 after parameter optimization. This ensures there is genuine economic alpha in the training period, not just statistical cointegration. Pairs that are cointegrated but not profitable (e.g., AT&T/Verizon — structural decline in AT&T) are correctly eliminated.

---

## 3. Adaptive Filter Theory

### 3.0 Shared Linear Model

All filter implementations share the same **linear prediction model**:

```
ŷ[t]  = w[t]ᵀ · x[t]              (prediction at time t)
e[t]  = y[t] − ŷ[t]               (prediction error = spread)
w[t+1] = w[t] + f(e[t], x[t], ·)  (weight update rule)
```

With `n_taps = 1`, `w[t]` is a single scalar — the hedge ratio `β[t]`.

---

### 3.1 The NLMS Filter — Foundation

The **Normalized Least Mean Squares (NLMS)** filter normalizes the LMS gradient update by the instantaneous input power:

```
w[t+1] = w[t] + μ · e[t] · x[t] / (‖x[t]‖² + ε)
```

The `‖x[t]‖² + ε` term makes the effective step size independent of input amplitude. This normalization has a clean geometric interpretation: the update projects the weight vector onto the hyperplane `{w : wᵀx[t] = y[t]}`, moving a fraction μ toward it.

**The fundamental NLMS dilemma**: large μ converges fast but tracks noise; small μ is stable but slow to adapt. A fixed μ cannot be optimal for all market regimes.

---

### 3.2 VS-NLMS — Variable Step-Size (Production Filter)

**Variable Step-Size NLMS** (Kwong & Johnston, 1992) adapts μ online using the **sign correlation** between consecutive prediction errors:

```
corr[t]  =  sign(e[t]) · sign(e[t-1])     ∈ {−1, 0, +1}
μ[t]     =  clip( α · μ[t-1]  +  γ · corr[t],   μ_min,  μ_max )
w[t+1]   =  w[t]  +  μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)
```

| `corr[t]` | Meaning | Response |
|---|---|---|
| `+1` | Errors same sign → filter systematically wrong | Increase μ: adapt faster |
| `−1` | Errors alternate → filter oscillating around solution | Decrease μ: stabilize |
| `0` | One error is zero | No change |

**Why use `sign()` rather than the raw product?** The `sign()` function makes the rule scale-invariant (independent of price levels) and robust to outliers.

**Optimized parameters**:

| Parameter | Value | Meaning |
|---|---|---|
| `mu_init` | 0.01–0.20 | Starting μ (grid-searched per fold) |
| `mu_min` | 0.001 | Floor |
| `mu_max` | 0.50 | Cap: prevents instability |
| `alpha` | 0.990 | Momentum of μ |
| `gamma` | 0.050 | Scale of adjustment per step |

---

## 4. The VS-NLMS Filter in Detail

### 4.1 β[t] IS the Filter Weight

```
Filter weight w[t]   ←→   Hedge ratio β[t]
Filter input x[t]    ←→   price_X[t]
Filter target y[t]   ←→   price_Y[t]
Filter error e[t]    ←→   Spread: price_Y[t] − β[t]·price_X[t]
```

The filter IS the hedge ratio estimator. Prediction errors ARE the spread.

### 4.2 Why μ ~ 0.43 Emerges as Optimal

In classical DSP, NLMS typically uses μ ∈ [0.01, 0.2]. For most pairs in this universe, the walk-forward grid search and the VS-NLMS adaptation converge toward μ ≈ 0.43 — much higher than intuition suggests.

The reason: hedge ratios have been in **persistent multi-month trends**. When β drifts upward over weeks, consecutive spread errors consistently have the same sign. The VS-NLMS rule correctly identifies this as chronic underfitting (`corr = +1` repeatedly) and pushes μ high to track the drift aggressively.

### 4.3 ML vs Baseline — When Each Wins

| Signal quality | Baseline (heuristic) | ML (MLP) | Why |
|---|---|---|---|
| Very clean (Nestle/Unilever) | **Wins** | Loses | Spread so stable that Kwong-Johnston already optimal; ML adds noise |
| Moderate (V/MA, MCO/SPGI) | Loses | **Wins** | Regime changes present; ML detects when to adapt faster/slower |
| Noisy (few folds, low Sharpe) | Both poor | Both poor | No consistent signal to learn |

---

## 5. Trading Strategy

### 5.1 Z-Score Computation

The raw spread `e[t]` is normalized using a rolling z-score:

```
μ_s[t]  = mean(e[t−L : t])
σ_s[t]  = std(e[t−L : t])
z[t]    = (e[t] − μ_s[t]) / σ_s[t]
```

Lookback `L` is optimized per fold (30–120 days in the grid).

### 5.2 Signal Generation

```
z[t] < −entry_z  →  signal[t+1] = +1   (spread below band: long Y, short X)
z[t] > +entry_z  →  signal[t+1] = −1   (spread above band: short Y, long X)
|z[t]| < exit_z  →  signal[t+1] =  0   (spread reverted: close position)
```

Signals use `t+1` to avoid lookahead: observe z-score at close on day t, act at open on day t+1.

**Optimized thresholds**: `entry_z = 1.5σ`, `exit_z = 0.25σ – 0.75σ` (varies by fold).

### 5.3 Z-Score Proportional Position Sizing

Position size scales proportionally to z-score magnitude:

```
position_size[t] = signal_direction[t] · |z[t]| / entry_z
```

A spread at 3σ gets a 2× larger position than one at 1.5σ. This is a mild Kelly-type sizing that naturally increases exposure when conviction is higher.

### 5.4 Return Calculation

```
spread_return[t]   = ΔY[t]/Y[t-1] − β[t-1] · ΔX[t]/X[t-1]
strategy_return[t] = signal[t-1] · spread_return[t]
```

Where `β[t-1]` is the previous day's hedge ratio (no lookahead).

---

## 6. Walk-Forward Validation Methodology

### 6.1 Why Walk-Forward?

Simple in-sample optimization captures idiosyncrasies of the historical period — this is **selection bias**. Walk-forward ensures every test data point is evaluated after the model was trained exclusively on data from the past relative to that point.

### 6.2 Implementation

```
Train window:  756 days (~3 years)
Test window:   252 days (~1 year)
Step:          252 days (non-overlapping test periods)
Total folds:   13 (for 16 years of data)

Fold 1:  Train [2010 → 2013]  →  Test [2013 → 2014]
Fold 2:  Train [2011 → 2014]  →  Test [2014 → 2015]
...
Fold 13: Train [2022 → 2025]  →  Test [2025 → 2026]
```

Train windows overlap. The critical invariant: **test windows never overlap**, and no future information leaks into training.

### 6.3 What is Optimized Per Fold

| Parameter | Grid | Notes |
|---|---|---|
| `mu_init` | [0.01, 0.05, 0.10, 0.20] | Initial μ for VS-NLMS |
| `lookback` | [30, 60, 90, 120] days | Z-score rolling window |
| `entry_z` | [1.5] | Fixed (best in most folds) |
| `exit_z` | [0.25, 0.50, 0.75] | Exit threshold |

**Total**: 48 combinations per fold. **Objective**: `score = Sharpe + 0.5 × MaxDrawdown`.

### 6.4 Filter Warm-Up

The VS-NLMS filter is run on the full window (train + test) during evaluation, but only the test period is recorded. This warm-up ensures the filter has converged to a stable hedge ratio before the test period begins.

### 6.5 Overfitting Detection

```
degradation = 1 − (avg_test_sharpe / avg_train_sharpe)
```

Degradation below 30% is acceptable. Above 50% is a warning sign.

---

## 7. Universe Scan Results — All Pairs Tested

Over 70 pairs tested across equities (US, European, Asian ADRs), FX, and bonds. The **three-gate filter** (Johansen + half-life + train Sharpe) eliminates the vast majority.

### 7.1 Cointegration Rate by Pair — Summary

| Sector | Best pair | Coint. folds | % | Result |
|--------|-----------|-------------|---|--------|
| Intl FMCG | Nestle/Unilever (NSRGY/UNLYF) | 8/12 | 67% | ✓ EXCELENTE |
| SG Banks | DBS/UOB (DBSDY/UOVEY) | 8/13 | 62% | ✓ BUENO |
| US Pagos | Visa/Mastercard (V/MA) | 5/14 | 36% | ✓ BUENO |
| Ratings | Moody's/S&P Global (MCO/SPGI) | 4/13 | 31% | ✓ BUENO |
| Lujo | LVMH/Kering (LVMUY/PPRUY) | 3/11 | 27% | ~ OK |
| Spirits | Diageo/Pernod (DEO/PRNDY) | 5/13 | 38% | ~ OK |
| Intl Pharma | Roche/AstraZeneca (RHHBY/AZN) | 4/13 | 31% | ~ OK |
| Intl Energía | BP/Shell (BP/SHEL) | 3/13 | 23% | ~ marginal |
| US Banca | Goldman/Morgan Stanley (GS/MS) | 1/13 | 8% | ~ marginal |

### 7.2 Pairs That Failed (representative sample)

The following pairs showed < 15% cointegration across all folds, making them untradeable with this methodology:

- **US Technology**: AMD/Intel, MSFT/Oracle, QCOM/Broadcom, ASML/AMAT — strong bull market trends break cointegration
- **US Financials**: JPM/BAC, BLK/SCHW, USB/PNC, STT/BK — structural differences post-2008 regulation
- **US Healthcare**: JNJ/ABT, PFE/MRK, UNH/CVS — M&A activity and pipeline divergence
- **US Industrials**: CAT/DE, HON/3M, UPS/FDX, BA/LMT — too different business models
- **US Energy**: XOM/CVX, SLB/HAL, COP/MRO — commodity cycles dominate over mean-reversion
- **Telecom**: T/VZ — AT&T structural decline (TimeWarner debt) permanently broke the pair
- **Asian markets**: Toyota/Honda, Alibaba/JD, TSMC/UMC, Infosys/Wipro — regulatory risk and regime changes
- **Japanese insurance/banks**: Nomura/Mizuho, Sumitomo/Mizuho — 23% max, BOJ policy distortions
- **Australian banks**: CBA/NAB — Sharpe 0.11, insufficient signal

**Root cause**: the 2010–2026 US bull market produces high cross-sectional correlation but not cointegration. Spreads trend rather than revert. Only pairs with truly identical economics (payment duopolies, FMCG giants, financial data oligopolies) maintain long-run equilibrium.

---

## 8. Best Pairs — Detailed Results

All results use `--no-background` (each fold trained only on its own data, no pool leakage) and the three-gate filter cascade.

| Par | Tickers | Sector | Folds | Sharpe ML | Sharpe Base | Return ML (ann) | Max DD | Operable now | ML wins |
|-----|---------|--------|-------|-----------|-------------|-----------------|--------|--------------|---------|
| Nestle / Unilever | NSRGY / UNLYF | Intl FMCG | 8/12 | 2.36 | **2.64** | 41.6% | −13% | ✗ | 1/8 |
| Visa / Mastercard | V / MA | US Pagos | 5/14 | **1.09** | 0.81 | 14.9% | −15% | ✗ | 3/5 |
| LVMH / Kering | LVMUY / PPRUY | Lujo | 3/11 | **0.92** | 0.61 | 13.1% | −12% | ✗ | 3/3 |
| Moody's / S&P Global | MCO / SPGI | Ratings | 4/13 | **0.91** | 0.85 | 9.4% | −7% | **✓** | 3/4 |
| DBS / UOB | DBSDY / UOVEY | SG Bancos | 8/13 | **0.91** | 0.77 | 8.6% | −14% | ✗ | 3/8 |
| Diageo / Pernod | DEO / PRNDY | Spirits | 5/13 | 0.62 | **0.93** | 6.2% | −14% | ✗ | 0/5 |
| Roche / AstraZeneca | RHHBY / AZN | Intl Pharma | 4/13 | 0.57 | **0.68** | 16.4% | −28% | ✗ | 1/4 |
| BP / Shell | BP / SHEL | Intl Energía | 3/13 | **0.59** | 0.53 | 13.4% | −16% | ✗ | 2/3 |
| Goldman / Morgan Stanley | GS / MS | US Banca | 1/13 | **0.70** | 0.58 | 8.3% | −10% | ✗ | 1/1 |

**Notes:**
- **Operable now**: the most recent fold (train 2021–2024, test 2024–2025) passed all three gates
- **Baseline wins** when the pair's signal is very clean (Nestle, Diageo) — the Kwong-Johnston heuristic already optimal
- **ML wins** when moderate signal with regime changes (V/MA, MCO/SPGI, LVMH/Kering)
- MCO/SPGI is the **only currently operable pair** with Sharpe > 0.9 and DD < −8%
- Roche/AZN has the highest annualized return (16.4%) but also the highest drawdown (−28%)

---

## 9. Macro Universe — FX, Bonds, Commodities and Indices

### 9.1 Motivation

Equity pairs trading relies on shared sector exposure. Macro pairs trading exploits a different class of relationships: **structurally anchored economic linkages** — currency blocs tied to the same commodity, yield curves within the same sovereign, developed vs. emerging market equity cycles.

These relationships differ fundamentally from sector pairs:
- Driven by global macro regimes (risk-on/risk-off, commodity supercycles, central bank policy divergence) rather than company-level fundamentals
- More stable long-term but subject to **hard regime breaks** (e.g., SNB peg removal 2015, 2022 rate shock)
- Require different data sources: FX via yfinance (expressed as USD per unit of foreign currency), bond yields via FRED (converted to zero-coupon prices), indices via ETFs

### 9.2 Data Sources

| Asset class | Source | Ticker format | Notes |
|---|---|---|---|
| FX pairs | yfinance | `XXXUSD=X` | USD per unit of foreign currency |
| Bond yields | FRED | `DGS2`, `DGS10` | Converted to price: P = exp(−y·T) |
| Equity indices | yfinance | ETF tickers | EFA (MSCI EAFE), EEM (MSCI EM) |
| Commodities | yfinance | Spot/ETF | GLD, SLV, USO, BNO... |

FX prices are expressed as USD per unit of foreign currency: NOKUSD=X rising means NOK strengthening. Long NOK / short CAD = bet on Norwegian krone outperforming Canadian dollar.

### 9.3 The nlms-macro Parallel Project

Macro assets are handled in the `nlms-macro/` parallel project, which uses the same walk-forward engine, three-gate filter cascade, and VS-NLMS/ML infrastructure — applied to non-equity time series.

```
nlms-macro/
├── walk_forward_ml_mu.py   # Same engine as nlms-zsizing
├── fetch_yields.py         # Downloads DGS2/5/10/30 from FRED
├── fetch_fx.py             # Downloads FX pairs from yfinance
└── data/
    ├── nok_cad.csv         # NOK/CAD FX pair (best FX result)
    ├── aud_cad.csv         # AUD/CAD (commodity currencies)
    ├── eur_chf.csv         # EUR/CHF (SNB-anchored)
    ├── efa_eem.csv         # EFA/EEM (DM vs EM equity indices)
    └── 2y10y.csv           # 2Y/10Y yield spread
```

### 9.4 Macro Pairs Tested — Full Results

Over 16 macro asset classes tested. The same three-gate filter (Johansen + half-life + train Sharpe) applies.

**Tradeable pairs found:**

| Pair | Tickers | Class | Folds | Sharpe ML | Sharpe Base | Return (ann) | Max DD | ML wins |
|------|---------|-------|-------|-----------|-------------|--------------|--------|---------|
| EFA / EEM | EFA / EEM | Indices | 4/13 | **1.32** | 0.83 | 9.3% | −7.2% | 2/4 |
| NOK / CAD | NOKUSD=X / CADUSD=X | FX | 6/13 | 0.72 | 0.72 | 13.3% | −10.9% | 4/6 |
| AUD / CAD | AUDUSD=X / CADUSD=X | FX | 3/13 | **0.61** | 0.50 | 3.9% | −5.1% | 1/3 |
| EUR / CHF | EURUSD=X / CHFUSD=X | FX | 5/13 | 0.51 | 0.51 | 7.1% | −8.2% | 3/5 |
| 2Y / 10Y spread | DGS2 / DGS10 | Bonds | 1/13 | 0.58 | 0.40 | 2.1% | −2.3% | 1/1 ⚠ |

⚠ Only 1 tradeable fold — result not statistically reliable.

**Pairs that failed (0 tradeable folds or untradeable drawdown):**

| Pair | Failure reason |
|---|---|
| Gold / Silver (GLD/SLV) | Only 15% cointegration — silver's industrial demand diverges in equity crises |
| WTI / Brent (USO/BNO) | 23% cointegration — futures ETF roll costs create structural drift |
| Nat Gas / Oil | Cointegrated but explosive drawdown (−83%) — gas price regime breaks |
| LQD / HYG (IG vs HY bonds) | 0% cointegration — credit spread is non-stationary over 16 years |
| SPY / TLT (equity vs bonds) | 0% cointegration — risk-on/off regime changes are permanent breaks |
| SPY / IWM, SPY / QQQ | 0% cointegration — factor tilts diverge structurally |
| JPY / CHF | Cointegrated but 0 tradeable folds — BoJ yield curve control creates non-stationarity |
| Agricultural (corn/wheat, soy/corn) | 0% cointegration — storage costs and seasonal patterns dominate |

### 9.5 Key Insights

**Why EFA/EEM works best**: Developed and emerging market equity indices share global growth exposure but diverge in risk-off cycles. The spread captures the **DM/EM premium cycle**, which has multi-year regimes that the ML filter detects better than the heuristic — explaining the large ML vs. baseline gap (1.32 vs. 0.83).

**Why commodity pairs fail**: Futures ETFs carry **roll costs** that produce structural drift in the spread. Gold/Silver fails despite the well-known "gold/silver ratio" — silver's industrial component creates divergences that don't revert on a 3-year horizon.

**Why FX pairs work moderately**: FX pairs share underlying commodity/trade linkages but exhibit **regime breaks** from central bank interventions and peg changes. NOK/CAD is the most stable because both currencies float freely and are anchored to Brent vs. WTI crude — two highly correlated benchmarks with minimal structural basis risk.

**Why bond pairs mostly fail**: Bond ETFs have duration that drifts as rates change, making the relative value non-stationary. The only marginal result (2Y/10Y) has just 1 tradeable fold — insufficient for live deployment.

---

## 10. Machine Learning Experiments

### 10.1 ML_VSNLMSFilter — Replacing the μ Heuristic

**Motivation**: The Kwong-Johnston rule is a hand-crafted heuristic. Can a neural network learn a superior mapping from error history to optimal μ?

**Implementation**: `src/ml_nlms.py` — `train_mu_predictor()` and `ML_VSNLMSFilter`.

#### Training Phase (per fold, on training data only)

**Step 1 — Parallel filter competition**: Run 9 NLMS filters simultaneously, each with a different fixed μ from `[0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]`.

**Step 2 — Target generation (mean-reversion criterion)**:

```
acorr_i[t] = corr(e_i[t−20:t−1], e_i[t−19:t])   for each filter i
winner[t]  = argmin_i  acorr_i[t]                  (most mean-reverting spread)
Y[t]       = μ_candidates[winner[t]]               (target μ for the MLP)
```

**Step 3 — Feature extraction** (12 features per timestep):

| Feature | Description |
|---|---|
| z-score rolling | Current spread z-score (5-day window) |
| Autocorrelation | Lag-1 autocorr of spread (5-day) |
| Spread volatility | Rolling std of spread |
| Sign correlation | `sign(e[t])·sign(e[t-1])` |
| μ current | Current VS-NLMS step size |
| β current | Current hedge ratio estimate |
| + 6 lag features | Features at t-1, t-2, t-5 |

**Step 4**: Train `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)` to predict optimal μ from these features.

### 10.2 --no-background Flag

Each fold trains the ML model only on its own training data. Without this flag, all previous folds' data accumulates in a shared pool, creating **temporal contamination**: the model trained on fold 10 has implicitly seen the future (folds 1-9 include years closer to test periods of earlier folds).

```bash
# Correct: no pool contamination
python walk_forward_ml_mu.py --data=data/V_MA.csv --pair="V/MA" --no-background

# Incorrect: pool grows across runs, Sharpe inflated
python walk_forward_ml_mu.py --data=data/V_MA.csv --pair="V/MA"
```

---

## 11. How to Use the Strategy in Practice

This section explains **how to actually run the strategy** — when to train, when to test, when to trade, and when to stop.

### 11.1 Overall Workflow

```
Every year (or when a fold expires):
  1. Download fresh data (3 years back + current year)
  2. Run the three gates on the new training window
  3. If all gates pass: run grid search, get best parameters
  4. Deploy the structural hyperparameters (lookback, entry_z, exit_z) and the frozen MLP model for the next 252 trading days. Note: the MLP predicts a new μ[t] every day based on current market features — the hedge ratio β[t] adapts continuously. What stays fixed is the model architecture and weights, not the step size.
  5. Monitor the live spread daily: generate entry/exit signals
  6. After 252 days: go back to step 1 (new fold)
```

### 11.2 Step-by-Step: Training a New Fold

**When to train**: once per year, at the end of the previous test period. Concretely, if your current test window expires on December 31st, retrain the weekend before January 1st.

**Data required**: 3 years of daily adjusted close prices for both tickers. Use `yfinance`:

```python
import yfinance as yf
df = yf.download(["MCO", "SPGI"], start="2022-01-01", end="2025-01-01",
                 auto_adjust=True)["Close"]
```

**Gate 1 — Johansen test**: run on the full 3-year window:
```python
from src.strategy import johansen_cointegration
result = johansen_cointegration(df["MCO"].values, df["SPGI"].values)
if not result["cointegrated"]:
    print("SKIP — pair not cointegrated this year")
    # Do NOT trade this fold
```

**Gate 2 — Half-life**:
```python
from src.strategy import compute_halflife
hl = compute_halflife(df["MCO"].values, df["SPGI"].values)
if not hl["in_range"]:  # must be [3, 60] days
    print(f"SKIP — half-life {hl['halflife']:.0f}d out of tradeable range")
```

**Gate 3 — Grid search + train Sharpe**:
```python
# Run walk_forward_ml_mu.py on the training period only
# Best parameters are saved to results/walk_forward_ml_mu_folds.csv
# Check that best_train_sharpe >= 0.30
```

If all three gates pass, extract the best parameters (lookback, entry_z, exit_z, mu_init) from the training optimization.

### 11.3 Step-by-Step: Generating Daily Signals

**Every trading day** (after market close, before next day's open):

1. **Download today's closing prices** for both tickers
2. **Update the VS-NLMS filter** with today's price:
   ```python
   beta_today = filter.step(price_x_today, price_y_today)
   spread_today = price_y_today - beta_today * price_x_today
   ```
3. **Update the rolling z-score** (using the lookback from training):
   ```python
   z_today = (spread_today - spread_mean_rolling) / spread_std_rolling
   ```
4. **Check signal**:
   ```python
   if z_today < -1.5 and current_position == 0:
       # OPEN LONG: buy Y, short X
       # Size: |z_today| / 1.5 × notional
   elif z_today > +1.5 and current_position == 0:
       # OPEN SHORT: sell Y, buy X
       # Size: |z_today| / 1.5 × notional
   elif abs(z_today) < exit_z and current_position != 0:
       # CLOSE: flatten both legs
   ```
5. **Execute at next day's open** (never at today's close — that is lookahead)

### 11.4 Position Sizing

The strategy is **dollar-neutral**: for every $1 of Y, short `β[t]` dollars of X.

```
notional_Y = total_capital × position_fraction × |z[t]| / entry_z
notional_X = notional_Y × β[t]

shares_Y = round(notional_Y / price_Y)
shares_X = round(notional_X / price_X)
```

**Recommended maximum position fraction**: 10–20% of capital per pair. If running multiple pairs simultaneously, allocate proportionally to Sharpe (e.g., MCO/SPGI at 0.91 gets a larger slice than BP/Shell at 0.59).

### 11.5 When to Stop Trading a Fold

Stop (close all positions and wait for the next annual retrain) if any of the following happen:

| Condition | Action |
|---|---|
| The fold's test period expires (252 days) | Retrain next fold |
| Max drawdown in OOS exceeds 2× the historical max DD | Stop and retrain immediately |
| The spread has been outside the bands for > 60 days continuously | The pair may have broken — stop |
| A major corporate event (M&A, spin-off, bankruptcy) for either ticker | Stop immediately, retrain after event |

### 11.6 Running the Provided Scripts

```bash
# 1. Download data for a specific pair
python scan_universe.py --sector financials   # runs all financials sector pairs

# 2. Full walk-forward for a specific pair (recommended: --no-background)
python walk_forward_ml_mu.py --data=data/scan/mco_spgi.csv \
                              --pair="MCO / SPGI" \
                              --no-background

# 3. Scan the full universe (all 70+ pairs, takes ~30 min)
python scan_universe.py

# 4. Check current operability of all best pairs
# Look for the most recent fold in results/walk_forward_ml_mu_folds.csv
# If the last fold passed all three gates → pair is operable now
```

### 11.7 Currently Operable Pairs (as of March 2026)

| Par | Sharpe ML | Action |
|-----|-----------|--------|
| **MCO / SPGI** | **0.91** | ✓ Trade now — last fold passed all gates |
| V / MA | 1.09 | ✗ Last fold borderline (train Sharpe 0.50) — monitor |
| DBS / UOB | 0.91 | ✗ Last fold failed Johansen (trace=13.8) — monitor |

**Monitor** means: run the Johansen test weekly on the last 756 days. If trace > 15.5 for two consecutive weeks → consider entering.

---

## 12. Project Structure

```
nlms-zsizing/
│
├── walk_forward_ml_mu.py   # Main entry point — walk-forward baseline vs MLP
├── scan_universe.py        # Universe scanner — tests 70+ pairs automatically
├── portfolio_sharpe.py     # Portfolio Sharpe from 11 OOS return series
├── walk_forward.py         # Simpler walk-forward (baseline VS-NLMS only)
├── scan_pairs.py           # Lightweight pair scanner
├── analyze_pairs.py        # Post-scan analysis utilities
│
├── src/
│   ├── nlms.py             # VSNLMSFilter — core adaptive filter (Kwong-Johnston)
│   ├── ml_nlms.py          # ML_VSNLMSFilter — MLP μ predictor + training pipeline
│   ├── gru_nlms.py         # GRU_VSNLMSFilter — GRU μ predictor (--gru flag)
│   ├── cointegration.py    # Johansen test + half-life estimator
│   ├── signals.py          # Z-score computation + signal state machine
│   ├── optimizer.py        # Grid search + run_filter_pipeline
│   ├── backtest.py         # P&L simulator
│   ├── model_store.py      # Persistent MLP training pool (models/)
│   ├── strategy.py         # High-level strategy wrapper
│   ├── plots.py            # Equity curve + spread visualisation
│   └── data_generator.py   # Synthetic cointegrated pair generator (for tests)
│
├── data/                   # CSVs — NOT committed (downloaded via yfinance)
│   ├── visa_mastercard.csv # Primary development pair
│   ├── scan/               # 70+ pairs scanned from the universe
│   └── .gitkeep
│
├── models/                 # Accumulated MLP training data — NOT committed
│   └── .gitkeep
│
├── results/                # Walk-forward outputs — NOT committed
│   └── .gitkeep
│
├── ibex35_statistical_arbitrage.ipynb  # Reference notebook (third-party)
├── requirements.txt
├── README.md               # English documentation
└── README_ES.md            # Spanish documentation
```

---

## 13. Installation and Usage

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/nlms-pairs-trading.git
cd nlms-pairs-trading/nlms-zsizing
pip install -r requirements.txt

# 2. Download data for a pair (examples)
python - <<'EOF'
import yfinance as yf, pandas as pd
d1 = yf.download("V",  start="2010-01-01", end="2026-01-01", auto_adjust=True)["Close"].squeeze()
d2 = yf.download("MA", start="2010-01-01", end="2026-01-01", auto_adjust=True)["Close"].squeeze()
df = pd.DataFrame({"price_x": d1, "price_y": d2}).dropna().reset_index()
df.columns = ["date","price_x","price_y"]
df.to_csv("data/visa_mastercard.csv", index=False)
EOF

# 3. Run walk-forward — baseline VS-NLMS vs MLP (recommended: --no-background)
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --no-background

# Use GRU instead of MLP
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --no-background --gru

# 4. Scan the full equity universe (downloads + tests 70+ pairs)
python scan_universe.py

# 5. Compute portfolio Sharpe across all active pairs
python portfolio_sharpe.py
```

### Flags

| Flag | Description |
|------|-------------|
| `--data=path` | CSV file with columns `date, price_x, price_y` |
| `--pair="X/Y"` | Display name shown in output tables |
| `--no-background` | Train only on current fold data — no cross-pair leakage (recommended) |
| `--gru` | Use GRU_VSNLMSFilter instead of MLP_VSNLMSFilter |
| `--train=N` | Training window in days (default: 756) |
| `--test=N` | Test window in days (default: 252) |

---

## 14. Portfolio Construction and Results

### 14.1 Methodology

Individual Sharpe ratios cannot be averaged to obtain the portfolio Sharpe. The correct approach is:

1. **Extract the OOS daily return series** for each pair from the walk-forward MLP run (`--no-background`). Days where the fold is inactive (failed cointegration, half-life, or train Sharpe gate) are kept as zero — the capital is parked.
2. **Merge all series by date** into a single DataFrame, filling `NaN` with 0 on days where a pair has no active fold.
3. **Equal-weight average** across all N pairs: `portfolio[t] = mean(returns[t])`.
4. **Compute Sharpe on the combined series** — this correctly penalises inactive days.

The diversification benefit arises because the spread returns of structurally different pairs (different sectors, geographies, asset classes) are nearly uncorrelated. When correlations are close to zero, portfolio volatility falls proportionally to `1/√N` while expected return stays constant, boosting the Sharpe ratio.

### 14.2 Universe

11 pairs included (9 equity + 2 macro). WTI/Brent excluded (illiquid ETF proxies, structural roll costs).

| Pair | Asset class | Active folds | Sharpe (individual) |
|------|-------------|--------------|---------------------|
| V / MA | Payments | 4/13 | 1.30 |
| NSRGY / UNLYF | Consumer staples | 8/13 | 2.36 |
| DEO / PRNDY | Beverages | 5/13 | 0.62 |
| DBS / UOB | Singapore banks | 8/13 | 0.77 |
| MCO / SPGI | Ratings agencies | 4/13 | 0.91 |
| LVMUY / PPRUY | Luxury goods | 3/13 | 0.92 |
| RHHBY / AZN | Pharma | 4/13 | 0.57 |
| GS / MS | US investment banks | 1/13 | 0.70 |
| BP / SHEL | Energy majors | 3/13 | 0.59 |
| EFA / EEM | EM/DM indices | 4/13 | 1.32 |
| NOK / CAD | FX | 6/13 | 0.72 |

### 14.3 Portfolio Results (Equal Weight, MLP, --no-background)

| Metric | Value |
|--------|-------|
| **Sharpe ratio** | **1.94** |
| Annualised return | 5.5% |
| Annualised volatility | 2.8% |
| Max Drawdown | −3.6% |
| Days with position | 1,972 / 3,375 |
| Avg pairwise correlation | −0.003 |
| Diversification gain vs avg individual | **+98%** |

Annual returns:

| Year | Return |
|------|--------|
| 2013 | +4.9% |
| 2014 | +4.0% |
| 2015 | +4.7% |
| 2016 | +9.9% |
| 2017 | +3.7% |
| 2018 | +5.1% |
| 2019 | +0.6% |
| 2020 | +9.7% |
| 2021 | +20.8% |
| 2022 | +2.4% |
| 2023 | +3.2% |
| 2024 | +4.5% |

No negative year in 12 years of OOS data.

### 14.4 Why the Portfolio Sharpe Exceeds Individual Sharpes

Average individual Sharpe (computed on active days only within each fold period): 0.98. Portfolio Sharpe: 1.94 — a **+98% diversification gain**.

The near-zero average pairwise correlation (−0.003) confirms that spread returns across different sectors and asset classes are essentially independent. This is theoretically expected: the spread of V/MA captures Visa-specific vs Mastercard-specific risk; the spread of EFA/EEM captures developed vs emerging market relative risk. These sources of risk are structurally unrelated.

**Key limitation**: results exclude transaction costs (bid-ask spread, borrow costs for short leg, commissions). For liquid pairs (V/MA, MCO/SPGI, EFA/EEM), impact would be modest. For OTC ADR pairs (NSRGY/UNLYF, DBSDY/UOVEY), slippage may be significant.

---

## 15. References

1. **Kwong, C.P. & Johnston, E.W. (1992)**. "A variable step size LMS algorithm." *IEEE Transactions on Signal Processing*, 40(7), 1633–1642. — Foundation of the VS-NLMS filter used throughout.

2. **Johansen, S. (1988)**. "Statistical analysis of cointegration vectors." *Journal of Economic Dynamics and Control*, 12(2-3), 231–254. — The cointegration test used in Gate 1.

3. **Engle, R.F. & Granger, C.W.J. (1987)**. "Co-integration and error correction: representation, estimation, and testing." *Econometrica*, 55(2), 251–276. — Foundational cointegration theory.

4. **Vidyamurthy, G. (2004)**. *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance. — Standard reference for pairs trading methodology.

5. **Chan, E. (2013)**. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. — Practical pairs trading implementation guidance.

6. **Haykin, S. (2002)**. *Adaptive Filter Theory* (4th ed.). Prentice Hall. — Comprehensive reference for NLMS, RLS, and adaptive filter theory.
