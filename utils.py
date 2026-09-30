import streamlit as st
import pandas as pd
import numpy as np
import os


@st.cache_data
def load_data():
    """Load the PSA CSV and reshape from wide to long format."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Major Crops- Volume of Production, by Region, Quarter, and Semester, 2010-2026.csv",
    )
    raw = pd.read_csv(csv_path, header=1)

    # Keep only quarterly columns (skip Semester / Annual aggregates)
    quarter_cols = [
        c
        for c in raw.columns
        if "Quarter" in c and "Semester" not in c and "Annual" not in c
    ]

    # Wide → long
    long = raw.melt(
        id_vars=["Crop", "Geolocation"],
        value_vars=quarter_cols,
        var_name="Period",
        value_name="Volume",
    )

    # Extract Year and Quarter number
    long["Year"] = long["Period"].str.extract(r"(\d{4})").astype(int)
    long["Quarter"] = long["Period"].str.extract(r"Quarter(\d)").astype(int)

    # Coerce non-numeric volumes ('..') to NaN, then drop them
    long["Volume"] = pd.to_numeric(long["Volume"], errors="coerce")
    long.dropna(subset=["Volume"], inplace=True)

    # Remove country-level aggregate row; keep individual regions only
    long = long[long["Geolocation"] != "PHILIPPINES"].copy()

    # Clean region names (strip leading "..")
    long["Region"] = long["Geolocation"].str.strip().str.lstrip(".")

    long.drop(columns=["Geolocation", "Period"], inplace=True)
    long.reset_index(drop=True, inplace=True)
    return long
