# Technical Notes — LATAM Inflation Forecasting

## Why Prophet + LSTM (not just one)?

Prophet and a stacked LSTM are complementary, not redundant:

- **Prophet** is interpretable and fast. It decomposes a time series into trend, seasonality, and regressor effects explicitly. It works well when data has clear seasonal patterns and smooth trends — which most LATAM CPI series do. It produces calibrated confidence intervals natively. The downside: it assumes the relationship between regressors and the target is additive and linear.

- **Stacked LSTM** can learn nonlinear, sequential interactions across many features simultaneously. It does not need to be told what seasonality looks like; it learns it from data. The pooled multi-country architecture also lets the model transfer dynamics across countries (e.g., Brazil's 1990s disinflation helping predict Argentina's 2025 disinflation). The downside: it is a black box, requires more data, and can overfit on short series.

Running both lets us answer a richer question: where does statistical structure-imposing (Prophet) beat pattern-matching (LSTM), and vice versa?

---

## Data source substitutions and why

| Indicator | Original plan | Actual source | Reason for substitution |
|---|---|---|---|
| Monthly CPI | World Bank `FP.CPI.TOTL` | OECD SDMX API | World Bank only provides annual CPI |
| Monthly CPI | IMF IFS SDMX | OECD SDMX API | `dataservices.imf.org` unreachable from build environment |
| MEX, BRA CPI | OECD `DF_PRICES_ALL` | OECD `DF_G20_PRICES` | Multi-country batch requests to `DF_PRICES_ALL` silently truncate; G20 dataset returns complete series for G20 members |
| CHL CPI 2024+ | OECD `DF_PRICES_ALL` | Splice with `DF_PRICES_C2018_ALL` | Chile switched from COICOP 1999 to COICOP 2018 methodology in 2024; both series use base year 2015=100 and produce identical values in the overlap — splice is lossless |
| COL, ARG FX | FRED `DEXCOUS`, `DEXARUS` | FRED `COLCCUSMA02STM`, `ARGCCUSMA02STM` | DEX series do not exist for COP and ARS; IFS Currency Conversions series provide equivalent monthly averages |
| COL policy rate | FRED `IRSTCI01COM156N` | FRED `COLIRSTCI01STM` | The OECD MEI series ID was wrong; call money rate is the closest monthly proxy |
| ARG policy rate | FRED | Not available | Argentina operates multiple simultaneous reference rates under capital controls; no single monthly "policy rate" series exists in FRED or OECD MEI |

---

## Why time-based splits matter for time series

Random train/test splits are standard for i.i.d. data (each sample is independent). Time series data is the opposite: each observation is correlated with its neighbors. If we shuffle randomly, a model trained on January 2023 data can "see" February 2023 in the test set — which means the test set is no longer a true holdout of the future.

The consequence is optimistically biased metrics. A model evaluated via random split may appear to achieve MAPE of 2%, but fail badly in production because it never actually learned to predict forward in time.

We use a strict chronological split throughout: the last 12 months of each country's data form the test set, and the model is trained only on data before that window. This mirrors the real-world use case (predicting next month's inflation from current and past data).

---

## How missing data was handled

| Data type | Treatment | Rationale |
|---|---|---|
| CPI index (MEX/BRA/CHL/COL) | Linear interpolation ≤ 3 months | CPI changes continuously; interpolation preserves the rate of change between known values better than forward-fill |
| FX rates | Forward-fill ≤ 2 months | The last known exchange rate is the correct prior for a short gap; markets or central banks set rates continuously |
| Policy rates | Forward-fill ≤ 3 months | Central banks hold rates between meetings; forward-fill is economically accurate, not just convenient |
| Argentina policy rate | Left as NaN | No reliable single-series proxy exists; imputing would introduce noise |
| Argentina CPI | Native YoY% from 2017-12 only | INDEC falsified data 2007–2016; OECD excludes that period |

---

## Argentina: a special case throughout

Argentina required country-specific handling at every phase:

1. **CPI data**: OECD excludes Argentina from full index datasets. Only post-2017 YoY% change data is available (85 months vs 360 for others), and only after INDEC resumed credible reporting.

2. **Policy rate**: Argentina's central bank (BCRA) operates multiple simultaneous rate instruments (repo rate, Leliq rate, overnight rate) alongside capital controls. No single monthly series reliably represents "the" policy rate. The feature is left as NaN for Argentina; the FX rate change carries much of the monetary signal instead.

3. **Official vs parallel FX rate**: The FRED series `ARGCCUSMA02STM` reflects Argentina's official exchange rate. Argentina maintained a "cepo cambiario" (exchange rate controls) for extended periods (2011–2015, 2019–2023), during which the official and parallel ("blue dollar") rates diverged by 50–100%. For modeling purposes, the official rate is used but this limitation is noted.

---

## Model performance and hypotheses

| Country | Prophet MAPE | LSTM MAPE | Better model | Hypothesis |
|---|---|---|---|---|
| Mexico | 11.0% | 12.0% | Prophet | Mexico's inflation is relatively stable and seasonal (food prices, energy subsidies). Prophet's explicit seasonality decomposition fits well. |
| Brazil | 12.9% | 5.0% | LSTM | Brazil had a complex, nonlinear disinflation arc 2022–2025. The LSTM's pooled model likely transferred dynamics from similar patterns in other countries. |
| Chile | 7.0% | 6.7% | Tie | Chile is the most macro-stable country in the set. Both models handle low-variance targets well. |
| Colombia | 12.2% | 21.3% | Prophet | Colombia shows irregular spikes (2022 food/energy shock) that Prophet handles via its regressor mechanism. The LSTM may have overfit to Brazil's smoother patterns at Colombia's expense. |
| Argentina | 192.6% | 36.2% | LSTM | Argentina's 2025 disinflation is a structural break with no precedent in 85 months of training data. The pooled LSTM borrowed from Brazil's 1990s disinflation (hyperinflation → stabilization) to partially capture this transition. Prophet, being country-specific, had no such prior. |
