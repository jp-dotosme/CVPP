# 🌾 Crop Volume of Production Predictor

A machine learning web application that predicts the production volume of major Philippine crops by **region** and **quarter**, built with Streamlit and scikit-learn.

---

## 📊 Project Overview

This project uses historical crop production data from the **Philippine Statistics Authority (PSA)** spanning 2010–2026 to train and compare three regression models:

| Model | Algorithm | R² Score | MAE (MT) |
|---|---|---|---|
| Linear Regression | `LinearRegression` | 0.0020 | 50,702.72 |
| Random Forest | `RandomForestRegressor` | 0.8232 | 7,663.55 |
| **XGBoost** ⭐ | `XGBRegressor` | **0.8731** | 13,040.69 |

**Selected model:** XGBoost (highest R² score of 0.8731).

---

## 📁 Folder Structure

```
CVPP/
├── app/                          # Streamlit application
│   ├── main.py                   # Landing / home page
│   ├── utils.py                  # Shared data loading utility
│   └── pages/
│       ├── 1_🌲_Random_Forest.py
│       ├── 2_📈_Linear_Regression.py
│       └── 3_🚀_XGBoost.py
├── data/
│   ├── original/                 # Raw dataset + source note
│   │   ├── Major Crops-....csv
│   │   └── source_note.txt
│   └── processed/                # Cleaned long-format dataset
│       └── crops_long_format.csv
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   └── 02_Model_Training_Comparison.ipynb
├── src/
│   └── preprocessing.py          # Reusable preprocessing functions
├── models/
│   ├── best_model.joblib         # Saved XGBoost model
│   ├── le_crop.joblib            # Crop label encoder
│   └── le_region.joblib          # Region label encoder
├── documentation/
│   └── technical_documentation.md
├── paper/                        # IMRaD paper (DOCX + PDF)
├── README.md                     # ← You are here
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CVPP.git
   cd CVPP
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```
   - Windows: `.\.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```
   The app will open at `http://localhost:8501`.

---

## 🚀 Usage

1. Open the app in your browser.
2. Use the **sidebar** to navigate between model pages.
3. Select a **Crop**, **Region**, **Year**, and **Quarter**.
4. The predicted production volume is displayed instantly along with a historical trend chart.

### Input Format

| Input | Type | Example |
|---|---|---|
| Crop | Dropdown selection | `Banana Cavendish` |
| Region | Dropdown selection | `REGION XI (DAVAO REGION)` |
| Year | Numeric (2010–2030) | `2026` |
| Quarter | Dropdown (1–4) | `3` |

### Output
- **Predicted Volume** in metric tons (MT)
- **Historical trend chart** with the prediction overlaid as a red diamond
- **Model performance metrics** (R² and MAE)
- **Feature importance** bar chart

---

## 📈 Dataset

- **Source:** [Philippine Statistics Authority — OpenSTAT](https://openstat.psa.gov.ph/)
- **Period:** 2010–2026 (quarterly)
- **Crops:** 57 major crops
- **Regions:** 17 Philippine administrative regions
- **Format:** CSV, wide format (~858 KB, reshaped to ~60K rows in long format)

See [`data/original/source_note.txt`](data/original/source_note.txt) for full details.

---

## 🤖 Models

All three models use the same preprocessing pipeline:
1. Reshape wide CSV → long format `(Crop, Region, Year, Quarter) → Volume`
2. Label-encode Crop and Region
3. 80/20 train-test split with `random_state=42`

### Limitations
- **Random Forest & XGBoost** do not extrapolate well beyond the training data range (2010–2026). Predictions for far-future years may plateau.
- **Label encoding** treats categories as ordinal, which is not ideal for linear models (hence Linear Regression's poor performance).
- The model does not account for external factors like weather, policy changes, or natural disasters.

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `streamlit: not recognized` | Run with `python -m streamlit run app/main.py` |
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install -r requirements.txt` |
| PowerShell execution policy error | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Port 8501 already in use | Kill the old process or use `streamlit run app/main.py --server.port 8502` |

---

## 📝 License

Dataset is sourced from Philippine government open data (public domain).
