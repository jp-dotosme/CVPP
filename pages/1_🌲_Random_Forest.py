import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import altair as alt

from utils import load_data


# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Random Forest – Crop Predictor",
    page_icon="🌲",
    layout="wide",
)


# ── Model Training ──────────────────────────────────────────────────
@st.cache_resource
def train_model(_df: pd.DataFrame):
    """Train a Random Forest Regressor and return model + encoders + metrics."""
    df = _df.copy()

    le_crop = LabelEncoder()
    le_region = LabelEncoder()
    df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
    df["Region_enc"] = le_region.fit_transform(df["Region"])

    features = ["Crop_enc", "Region_enc", "Year", "Quarter"]
    X = df[features]
    y = df["Volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    # Feature importance
    feat_importance = dict(zip(features, model.feature_importances_))

    return model, le_crop, le_region, metrics, feat_importance


# ── Load & Train ────────────────────────────────────────────────────
df = load_data()
model, le_crop, le_region, metrics, importances = train_model(df)


# ── Sidebar — User Inputs ───────────────────────────────────────────
st.sidebar.header("🔧 Prediction Input")

crop = st.sidebar.selectbox("🌱 Crop", sorted(df["Crop"].unique()))
region = st.sidebar.selectbox("📍 Region", sorted(df["Region"].unique()))
year = st.sidebar.number_input(
    "📅 Year", min_value=2010, max_value=2030, value=2026, step=1
)
quarter = st.sidebar.selectbox("📊 Quarter", [1, 2, 3, 4])


# ── Prediction ──────────────────────────────────────────────────────
crop_enc = le_crop.transform([crop])[0]
region_enc = le_region.transform([region])[0]

input_df = pd.DataFrame(
    {
        "Crop_enc": [crop_enc],
        "Region_enc": [region_enc],
        "Year": [year],
        "Quarter": [quarter],
    }
)
prediction = max(0, model.predict(input_df)[0])


# ── Main Layout ─────────────────────────────────────────────────────
st.title("🌲 Random Forest — Crop Volume Predictor")
st.caption(
    "Predict the production volume of major Philippine crops by region and quarter "
    "using a **Random Forest Regressor** trained on PSA data (2010–2026)."
)

# Prediction result
st.markdown("---")
pcol1, pcol2, pcol3, pcol4 = st.columns(4)
pcol1.metric("Crop", crop)
region_short = region.split("(")[0].strip() if "(" in region else region
pcol2.metric("Region", region_short[:25])
pcol3.metric("Period", f"{year} Q{quarter}")
pcol4.metric("Predicted Volume (MT)", f"{prediction:,.2f}")
st.markdown("---")


# ── Historical Trend Chart ──────────────────────────────────────────
st.subheader("📈 Historical Production Trend")

hist = df[(df["Crop"] == crop) & (df["Region"] == region)].copy()
hist = hist.sort_values(["Year", "Quarter"])

if not hist.empty:
    hist["Period_Label"] = (
        hist["Year"].astype(str) + " Q" + hist["Quarter"].astype(str)
    )
    hist["Time"] = hist["Year"] + (hist["Quarter"] - 1) / 4

    line_chart = (
        alt.Chart(hist)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time:Q", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("Volume:Q", title="Volume (MT)"),
            tooltip=[
                alt.Tooltip("Period_Label:N", title="Period"),
                alt.Tooltip("Volume:Q", title="Volume (MT)", format=",.2f"),
            ],
        )
        .properties(height=400)
        .interactive()
    )

    pred_point = pd.DataFrame(
        {
            "Time": [year + (quarter - 1) / 4],
            "Volume": [prediction],
            "Period_Label": [f"{year} Q{quarter} (Predicted)"],
        }
    )
    pred_chart = (
        alt.Chart(pred_point)
        .mark_point(size=200, color="red", shape="diamond", filled=True)
        .encode(
            x="Time:Q",
            y="Volume:Q",
            tooltip=[
                alt.Tooltip("Period_Label:N", title="Period"),
                alt.Tooltip("Volume:Q", title="Predicted (MT)", format=",.2f"),
            ],
        )
    )

    st.altair_chart(line_chart + pred_chart, use_container_width=True)
else:
    st.info(f"No historical data available for **{crop}** in **{region}**.")


# ── Model Performance ───────────────────────────────────────────────
st.subheader("🤖 Model Performance")

mcol1, mcol2 = st.columns(2)
mcol1.metric("R² Score", f"{metrics['r2']:.4f}", help="1.0 = perfect prediction")
mcol2.metric("Mean Absolute Error", f"{metrics['mae']:,.2f} MT")


# ── Feature Importance ──────────────────────────────────────────────
st.subheader("🔍 Feature Importance")

imp_df = (
    pd.DataFrame(
        {
            "Feature": ["Crop", "Region", "Year", "Quarter"],
            "Importance": [
                importances["Crop_enc"],
                importances["Region_enc"],
                importances["Year"],
                importances["Quarter"],
            ],
        }
    )
    .sort_values("Importance", ascending=False)
)

imp_chart = (
    alt.Chart(imp_df)
    .mark_bar()
    .encode(
        x=alt.X("Importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort="-x", title=""),
        color=alt.Color("Feature:N", legend=None),
    )
    .properties(height=200)
)
st.altair_chart(imp_chart, use_container_width=True)


# ── Raw Data Explorer ───────────────────────────────────────────────
with st.expander("📋 Browse Raw Data"):
    st.dataframe(df, use_container_width=True, height=400)
