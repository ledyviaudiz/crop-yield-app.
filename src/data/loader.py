# src/data/loader.py
# ============================================================
# DATA LOADING, COUNTRY HOLDOUT, SPLIT, AND ONE-HOT ENCODING
# ============================================================

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CO2_MAUNA_LOA, TIME_COL, ITEM_COL, MIN_YEARS_PER_AREA,
    TEST_YEAR_FRAC, HOLDOUT_COUNTRY_FRAC, SEED,
)

REQUIRED_RAW_COLUMNS = {
    "Area", "Item", "Year", "hg/ha_yield",
    "average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes",
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    missing_cols = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan pada yield_df.csv: {sorted(missing_cols)}")

    df = df.rename(columns={
        "hg/ha_yield": "crop_yield",
        "average_rain_fall_mm_per_year": "rainfall",
        "avg_temp": "temp_mean",
        "pesticides_tonnes": "pesticide_use",
    })
    print(f"[loader] Raw shape: {df.shape}")
    print(f"[loader] Raw countries: {df['Area'].nunique()} | crops: {df['Item'].nunique()} | years: {df['Year'].min()}-{df['Year'].max()}")
    return df


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    counts = df_raw.groupby("Area")[TIME_COL].nunique()
    valid = counts[counts >= MIN_YEARS_PER_AREA].index
    df = df_raw[df_raw["Area"].isin(valid)].copy()
    df = df.dropna(subset=["crop_yield", "rainfall", "temp_mean", "pesticide_use", "Area", ITEM_COL, TIME_COL])
    df = df.sort_values(["Area", ITEM_COL, TIME_COL]).reset_index(drop=True)

    print(f"[loader] Final analytical rows: {len(df):,}")
    print(f"[loader] Final countries: {df['Area'].nunique()} | crops: {df[ITEM_COL].nunique()}")
    print(f"[loader] Removed countries (<{MIN_YEARS_PER_AREA} years): {df_raw['Area'].nunique() - df['Area'].nunique()}")
    return df


def add_co2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co2_ppm"] = df[TIME_COL].map(CO2_MAUNA_LOA)
    if df["co2_ppm"].isna().any():
        year_to_co2 = pd.Series(CO2_MAUNA_LOA).sort_index()
        df["co2_ppm"] = df[TIME_COL].apply(
            lambda y: float(year_to_co2.iloc[np.argmin(np.abs(year_to_co2.index.to_numpy() - y))])
        )
    return df


def select_holdout_countries(df: pd.DataFrame, frac: float = HOLDOUT_COUNTRY_FRAC, seed: int = SEED):
    """Pilih negara holdout secara deterministik SEBELUM tuning/model selection."""
    countries = np.sort(df["Area"].unique())
    n_holdout = max(1, int(len(countries) * frac))
    rng = np.random.default_rng(seed)
    holdout = sorted(rng.choice(countries, size=n_holdout, replace=False).tolist())
    development = sorted(set(countries) - set(holdout))
    return development, holdout


def time_based_split(df: pd.DataFrame, eligible_mask=None, test_frac: float = TEST_YEAR_FRAC):
    """Split waktu hanya pada baris yang eligible (development countries)."""
    if eligible_mask is None:
        eligible_mask = np.ones(len(df), dtype=bool)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    years = np.sort(df.loc[eligible_mask, TIME_COL].unique())
    if len(years) < 2:
        raise ValueError("Minimal diperlukan dua tahun unik untuk time-based split.")
    cut = int(np.ceil(len(years) * (1 - test_frac)))
    cut = min(max(cut, 1), len(years) - 1)
    threshold_year = int(years[cut - 1])

    train_mask = eligible_mask & (df[TIME_COL].to_numpy() <= threshold_year)
    test_mask = eligible_mask & (df[TIME_COL].to_numpy() > threshold_year)
    print(f"[loader] Development time split: train <= {threshold_year} ({train_mask.sum():,}), test > {threshold_year} ({test_mask.sum():,})")
    return train_mask, test_mask, threshold_year


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")


def encode_onehot(df: pd.DataFrame, col: str, prefix: str, train_mask: np.ndarray):
    """Fit encoder pada development-training data, transform seluruh data."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:  # compatibility untuk sklearn lama
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

    ohe.fit(df.loc[train_mask, [col]])
    names = [f"{prefix}_{sanitize_name(c)}" for c in ohe.categories_[0]]
    if len(set(names)) != len(names):
        names = [f"{n}_{i}" for i, n in enumerate(names)]
    arr = ohe.transform(df[[col]])
    encoded = pd.DataFrame(arr, columns=names, index=df.index)
    out = pd.concat([df, encoded], axis=1)
    print(f"[loader] OHE {col}: {len(names)} columns, fitted on development-training only")
    return out, names, ohe


def encode_country_onehot(df: pd.DataFrame, train_mask: np.ndarray):
    return encode_onehot(df, "Area", "country", train_mask)


def encode_item_onehot(df: pd.DataFrame, train_mask: np.ndarray):
    return encode_onehot(df, ITEM_COL, "crop", train_mask)
