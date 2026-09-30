"""
Reusable preprocessing module for the Crop Volume Prediction project.

This module provides functions to load, clean, reshape, and encode
the PSA crop production dataset for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


CSV_FILENAME = "Major Crops- Volume of Production, by Region, Quarter, and Semester, 2010-2026.csv"


def find_csv(base_dir: str | None = None) -> str:
    """Locate the raw CSV file, searching common project paths."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = [
        os.path.join(base_dir, "data", "original", CSV_FILENAME),
        os.path.join(base_dir, CSV_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{CSV_FILENAME}' in any expected location."
    )


def load_raw(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw wide-format CSV (header on row 1, title on row 0)."""
    if csv_path is None:
        csv_path = find_csv()
    return pd.read_csv(csv_path, header=1)


def reshape_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the wide-format PSA dataframe into long format.

    Returns a DataFrame with columns:
        Crop, Region, Year, Quarter, Volume
    """
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

    # Coerce non-numeric volumes ('..') to NaN, then drop
    long["Volume"] = pd.to_numeric(long["Volume"], errors="coerce")
    long.dropna(subset=["Volume"], inplace=True)

    # Remove country-level aggregate; keep individual regions
    long = long[long["Geolocation"] != "PHILIPPINES"].copy()

    # Clean region names (strip leading "..")
    long["Region"] = long["Geolocation"].str.strip().str.lstrip(".")

    # Clean up
    long.drop(columns=["Geolocation", "Period"], inplace=True)
    long.reset_index(drop=True, inplace=True)

    return long


def encode_features(df: pd.DataFrame):
    """
    Label-encode Crop and Region columns.

    Returns:
        df_encoded: DataFrame with Crop_enc and Region_enc columns added
        le_crop: fitted LabelEncoder for Crop
        le_region: fitted LabelEncoder for Region
    """
    df_encoded = df.copy()
    le_crop = LabelEncoder()
    le_region = LabelEncoder()

    df_encoded["Crop_enc"] = le_crop.fit_transform(df_encoded["Crop"])
    df_encoded["Region_enc"] = le_region.fit_transform(df_encoded["Region"])

    return df_encoded, le_crop, le_region


def load_and_preprocess(csv_path: str | None = None) -> pd.DataFrame:
    """Full pipeline: load → reshape → clean.  Returns long-format DataFrame."""
    raw = load_raw(csv_path)
    return reshape_to_long(raw)
