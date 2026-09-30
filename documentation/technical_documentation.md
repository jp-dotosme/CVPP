# Technical Documentation — Crop Volume of Production Predictor

## 1. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit App                      │
│                                                     │
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ Landing   │  │ Model     │  │ Model Pages    │  │
│  │ Page      │  │ Selection │  │ (RF / LR / XGB)│  │
│  │ (main.py) │  │ (sidebar) │  │                │  │
│  └───────────┘  └───────────┘  └────────────────┘  │
│                        │                            │
│                 ┌──────┴──────┐                     │
│                 │  utils.py   │                     │
│                 │ (data load) │                     │
│                 └──────┬──────┘                     │
│                        │                            │
│              ┌─────────┴─────────┐                  │
│              │  CSV Dataset      │                  │
│              │  (data/original/) │                  │
│              └───────────────────┘                  │
└─────────────────────────────────────────────────────┘
```

## 2. Data Pipeline

### 2.1 Raw Data (Wide Format)
- **Source:** PSA OpenSTAT
- **Shape:** 1,026 rows × 121 columns
- **Format:** Each row = (Crop, Region); columns = `{Year} Quarter{N}`, `{Year} Semester{N}`, `{Year} Annual`

### 2.2 Preprocessing Steps
1. **Load CSV** with `header=1` (row 0 is a title row)
2. **Select quarterly columns** only — filter column names containing "Quarter" but not "Semester" or "Annual"
3. **Melt (unpivot)** from wide to long format → `(Crop, Geolocation, Period, Volume)`
4. **Parse period** — extract Year (int) and Quarter (int) from strings like "2010 Quarter1"
5. **Clean volumes** — coerce `..` to NaN, drop NaN rows
6. **Filter regions** — remove `"PHILIPPINES"` national aggregate row
7. **Clean region names** — strip leading `..` dots from geolocation strings
8. **Label encode** — map Crop and Region strings to integers for model input

### 2.3 Processed Data (Long Format)
- **Shape:** ~60,210 rows × 5 columns
- **Columns:** `Crop`, `Region`, `Year`, `Quarter`, `Volume`
- **Saved to:** `data/processed/crops_long_format.csv`

### 2.4 Train/Test Split
- **Ratio:** 80% train / 20% test
- **Method:** `train_test_split(random_state=42)` — stratification not applied (regression task)
- **Leak prevention:** no future data leaks since split is random, not temporal

## 3. Models

### 3.1 Linear Regression
- **Library:** `sklearn.linear_model.LinearRegression`
- **Hyperparameters:** default (OLS)
- **Pros:** fast, interpretable coefficients
- **Cons:** assumes linear relationship; label-encoded categories treated as ordinal

### 3.2 Random Forest Regressor
- **Library:** `sklearn.ensemble.RandomForestRegressor`
- **Hyperparameters:**
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `random_state=42`
  - `n_jobs=-1`
- **Pros:** handles non-linearity, robust to outliers, provides feature importance
- **Cons:** slower to train, does not extrapolate beyond training range

### 3.3 XGBoost Regressor
- **Library:** `xgboost.XGBRegressor`
- **Hyperparameters:**
  - `n_estimators=300`
  - `max_depth=8`
  - `learning_rate=0.1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_alpha=0.1`, `reg_lambda=1.0`
  - `random_state=42`
  - `n_jobs=-1`
- **Pros:** best R² score, regularised, handles non-linearity
- **Cons:** more complex to tune, does not extrapolate well

## 4. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² (Coefficient of Determination) | 1 − SS_res / SS_tot | 1.0 = perfect; 0 = baseline mean; <0 = worse than mean |
| MAE (Mean Absolute Error) | mean(\|y − ŷ\|) | Average prediction error in metric tons |
| RMSE (Root Mean Squared Error) | √mean((y − ŷ)²) | Penalises large errors more than MAE |

## 5. Application Guide

### 5.1 Running Locally
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

### 5.2 Deployed Version
- **Platform:** Streamlit Community Cloud
- **URL:** *(to be filled after deployment)*

### 5.3 User Interface
1. **Sidebar** — select Crop, Region, Year, Quarter
2. **Prediction card** — shows crop, region, period, and predicted volume (MT)
3. **Historical chart** — interactive Altair line chart with prediction overlaid
4. **Model metrics** — R² and MAE displayed as Streamlit metric cards
5. **Feature importance** — horizontal bar chart
6. **Raw data browser** — expandable dataframe view

## 6. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.64.0 | Web application framework |
| pandas | 3.0.6 | Data manipulation |
| numpy | 2.5.3 | Numerical computing |
| scikit-learn | 1.9.1 | ML models & preprocessing |
| xgboost | 3.4.1 | Gradient boosting model |
| altair | 6.3.0 | Interactive charts |
| matplotlib | * | Static plots (notebooks) |
| seaborn | * | Statistical visualisations (notebooks) |
| joblib | * | Model serialisation |

## 7. Known Limitations

1. **No extrapolation** — tree-based models cannot predict trends beyond the 2010–2026 training window.
2. **Label encoding** — ordinal encoding of Crop/Region is suboptimal for Linear Regression (one-hot encoding would be better but increases dimensionality).
3. **No external features** — weather, market prices, policy changes, and natural disasters are not captured.
4. **Aggregated resolution** — regional quarterly totals obscure province-level and monthly dynamics.
5. **Sparse crop-region pairs** — some crops have zero production in many regions, which can bias predictions.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `streamlit: not recognized` | Scripts folder not in PATH | Use `python -m streamlit run app/main.py` |
| `ModuleNotFoundError` | Missing package | Run `pip install -r requirements.txt` |
| Slow first load | Model training on first page visit | Normal — cached after first run |
| Negative predictions shown as 0.00 | `max(0, prediction)` clamp | By design — volume cannot be negative |
| `FileNotFoundError: CSV not found` | Wrong working directory | Run from the project root folder |
