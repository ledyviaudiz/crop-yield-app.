# src/predict.py
# ============================================================
# INFERENCE HELPERS untuk Streamlit (tanpa dependensi Streamlit,
# supaya bisa diuji terpisah). Memakai fungsi feature engineering
# ASLI dari pipeline training, jadi hasilnya konsisten.
# ============================================================
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.config import BASE_DIR, DEPLOYMENT_DIR, ITEM_COL
from src.data.loader import add_co2, sanitize_name
from src.features.engineer import (
    build_fc1, build_fc2, build_fc3, build_fc4, build_fc5,
)

DATA_FILE = os.path.join(BASE_DIR, "yield_df.csv")
HIST_COLS = ["Year", "rainfall", "temp_mean", "pesticide_use"]


def load_artifacts():
    d = DEPLOYMENT_DIR
    model = joblib.load(os.path.join(d, "best_model.pkl"))
    country_enc = joblib.load(os.path.join(d, "country_onehot_encoder.pkl"))
    item_enc = joblib.load(os.path.join(d, "item_onehot_encoder.pkl"))
    with open(os.path.join(d, "feature_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    meta = {}
    mp = os.path.join(d, "metadata.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    return model, country_enc, item_enc, stats, meta


def load_history(path: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(path):  # app tetap jalan tanpa dataset (riwayat diisi manual)
        return pd.DataFrame(columns=["Area", ITEM_COL, "Year", "crop_yield", *HIST_COLS[1:]])
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={
        "hg/ha_yield": "crop_yield",
        "average_rain_fall_mm_per_year": "rainfall",
        "avg_temp": "temp_mean",
        "pesticides_tonnes": "pesticide_use",
    })
    return df.sort_values(["Area", ITEM_COL, "Year"]).reset_index(drop=True)


def model_feature_names(model):
    for attr in ("feature_name_", "feature_names_in_"):
        v = getattr(model, attr, None)
        if v is not None:
            return list(v)
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        return list(booster().feature_names)
    raise RuntimeError("Tidak bisa membaca daftar fitur dari model.")


def _onehot(enc, col, prefix, value):
    cats = list(enc.categories_[0])
    names = [f"{prefix}_{sanitize_name(c)}" for c in cats]
    if len(set(names)) != len(names):  # sama persis dengan loader.encode_onehot
        names = [f"{n}_{i}" for i, n in enumerate(names)]
    out = {n: 0.0 for n in names}
    if value in cats:
        out[names[cats.index(value)]] = 1.0
    return out, (value in cats)


def build_input_row(area, item, year, rainfall, temp, pesticide, history, stats,
                    country_enc, item_enc):
    """history: DataFrame kolom HIST_COLS (tahun-tahun SEBELUM `year`, urut apa saja)."""
    hist = history.copy() if history is not None else pd.DataFrame(columns=HIST_COLS)
    hist = hist[HIST_COLS].dropna()
    hist = hist[hist["Year"] < year]
    new = pd.DataFrame([{
        "Year": year, "rainfall": rainfall, "temp_mean": temp, "pesticide_use": pesticide,
    }])
    df = pd.concat([hist, new], ignore_index=True)
    for c in HIST_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["Year"] = df["Year"].astype(int)
    df["Area"] = area
    df[ITEM_COL] = item
    df = df.sort_values("Year").reset_index(drop=True)
    df = add_co2(df)

    df = build_fc1(df)
    df = build_fc2(df, stats)
    df = build_fc3(df, stats)
    df = build_fc4(df, stats)
    df = build_fc5(df, stats)
    eng = [c for c in df.columns if c.startswith("fc")]
    df[eng] = df[eng].replace([np.inf, -np.inf], np.nan).fillna(0)

    row = df.iloc[[-1]].reset_index(drop=True)  # baris tahun target (Year terbesar)
    c_vec, c_known = _onehot(country_enc, "Area", "country", area)
    i_vec, i_known = _onehot(item_enc, ITEM_COL, "crop", item)
    enc = pd.DataFrame([{**c_vec, **i_vec}])
    return pd.concat([row, enc], axis=1), c_known, i_known


def predict_yield(model, full_row: pd.DataFrame) -> float:
    feats = model_feature_names(model)
    X = full_row.reindex(columns=feats, fill_value=0.0).astype(float)
    return float(model.predict(X)[0])
