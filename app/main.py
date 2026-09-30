import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Volume Production Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Home / Landing Page ─────────────────────────────────────────────
st.title("🌾 Crop Volume Production Predictor")
st.markdown(
    """
    Welcome! This app predicts the **volume of production** of major Philippine
    crops based on **region** and **quarter** using machine learning models
    trained on [PSA](https://openstat.psa.gov.ph/) data from 2010–2026.

    ---

    ### 📄 Pages

    Use the **sidebar** to navigate between the two models:

    | Page | Model | Description |
    |------|-------|-------------|
    | 🌲 **Random Forest** | `RandomForestRegressor` | An ensemble of 200 decision trees. Handles non-linear patterns and interactions well. |
    | 📈 **Linear Regression** | `LinearRegression` | A simple linear model. Fast and interpretable, but assumes a linear relationship. |
    | 🚀 **XGBoost** | `XGBRegressor` | Gradient-boosted trees. Often the most accurate for tabular data; learns from its own mistakes iteratively. |

    ---

    ### 🔧 How It Works

    1. The raw CSV (wide format) is **reshaped** into long format:
       `(Crop, Region, Year, Quarter) → Volume`
    2. Categorical features (Crop, Region) are **label-encoded**.
    3. The model is trained on **80 %** of the data and evaluated on the remaining **20 %**.
    4. Select a **Crop**, **Region**, **Year**, and **Quarter** in the sidebar
       of each page to get an instant prediction.

    ### 📊 Dataset

    - **Source:** Philippine Statistics Authority (PSA)
    - **Crops:** 57 major crops (rice, banana, mango, coffee, sugarcane, etc.)
    - **Regions:** 17 Philippine regions
    - **Period:** 2010–2026 (quarterly)

    👈 **Select a model page from the sidebar to get started!**
    """
)
