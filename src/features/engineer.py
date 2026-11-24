# src/features/engineer.py
# ============================================================
# LEAKAGE-CONTROLLED FEATURE CONSTRUCTION (FC1-FC5)
# ============================================================

import numpy as np
import pandas as pd

from src.config import FC1, FC2, FC3, FC4, FC5, ITEM_COL


def build_fc1(df: pd.DataFrame) -> pd.DataFrame:
    """Climate-derived indices from observed annual rainfall and mean temperature.

    `fc1_gdd` is a simplified annual thermal-excess proxy, not standard daily GDD.
    `fc1_pet` is a temperature-based PET proxy, not full monthly Thornthwaite PET.
    """
    df = df.copy()
    temp_pos = df["temp_mean"].clip(lower=0)
    df["fc1_gdd"] = (df["temp_mean"] - 10).clip(lower=0)
    df["fc1_ombrothermic"] = df["rainfall"] / (2 * df["temp_mean"].abs() + 1e-6)
    df["fc1_lang_rf"] = df["rainfall"] / (df["temp_mean"].abs() + 1e-6)
    df["fc1_htc"] = (df["rainfall"] * 10) / (df["fc1_gdd"] * 30 + 1e-6)
    df["fc1_aridity_dm"] = df["rainfall"] / (df["temp_mean"] + 10 + 1e-6)
    df["fc1_pet"] = 16 * (10 * temp_pos / 120).clip(lower=0) ** 1.514
    df["fc1_wbi"] = df["rainfall"] - df["fc1_pet"] * 12
    df["fc1_wsr"] = df["rainfall"] / (df["fc1_pet"] * 12 + 1e-6)
    return df


def fit_feature_stats(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    tr = df.loc[np.asarray(train_mask, dtype=bool)]
    if tr.empty:
        raise ValueError("Training subset kosong saat fit_feature_stats().")
    return {
        "pest_min": float(tr["pesticide_use"].min()),
        "pest_max": float(tr["pesticide_use"].max()),
        "pest_std": float(tr["pesticide_use"].std()),
        "year_min": int(tr["Year"].min()),
        "year_max": int(tr["Year"].max()),
        "thermal_proxy_max": float(max(tr["fc1_gdd"].max(), 1e-8)),
    }


def _pest_norm_feat(df: pd.DataFrame, stats: dict) -> pd.Series:
    rng = stats["pest_max"] - stats["pest_min"] + 1e-8
    return ((df["pesticide_use"] - stats["pest_min"]) / rng).clip(0, 1)


def _year_norm(df: pd.DataFrame, stats: dict) -> pd.Series:
    rng = max(stats["year_max"] - stats["year_min"], 1)
    return (df["Year"] - stats["year_min"]) / rng


def build_fc2(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    pest_norm = _pest_norm_feat(df, stats)
    df["fc2_pce"] = df["pesticide_use"] / (df["fc1_gdd"] + 1)
    df["fc2_pwi"] = df["pesticide_use"] / (df["rainfall"] + 1)
    opt_rain = (1 - np.abs(df["rainfall"] - 800) / 800).clip(0, 1)
    opt_temp = (1 - np.abs(df["temp_mean"] - 22) / 22).clip(0, 1)
    df["fc2_ocpi"] = (opt_rain + opt_temp) * pest_norm
    df["fc2_pipu"] = df["pesticide_use"] / (df["rainfall"] / 100 + 1)
    temp_opt = (1 - np.abs(df["temp_mean"] - 25) / 25).clip(0, 1)
    df["fc2_tpe"] = temp_opt * pest_norm
    co2_fert = np.log1p((df["co2_ppm"] - 354).clip(lower=0))
    df["fc2_cfwi"] = co2_fert * df["fc1_wsr"].clip(0, 3)
    return df


def build_fc3(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Strict-past temporal features per country-crop series."""
    df = df.sort_values(["Area", ITEM_COL, "Year"]).copy()
    grp = df.groupby(["Area", ITEM_COL], sort=False)["pesticide_use"]
    lag1 = grp.shift(1)
    lag2 = grp.shift(2)

    df["fc3_pest_lag1"] = lag1
    df["fc3_pest_lag2"] = lag2
    df["fc3_pest_roll3"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["fc3_pest_roll3_std"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    df["fc3_pest_delta"] = lag1 - lag2
    df["fc3_pest_pct_change"] = ((lag1 - lag2) / (lag2.abs() + 1e-8)).clip(-2, 2)

    yn = _year_norm(df, stats)
    df["fc3_year_sin"] = np.sin(2 * np.pi * yn)
    df["fc3_year_cos"] = np.cos(2 * np.pi * yn)
    decade_min = (stats["year_min"] // 10) * 10
    decade_max = max((stats["year_max"] // 10) * 10, decade_min + 10)
    df["fc3_decade_norm"] = (((df["Year"] // 10) * 10) - decade_min) / (decade_max - decade_min)

    rain_grp = df.groupby(["Area", ITEM_COL], sort=False)["rainfall"]
    df["fc3_rain_lag1"] = rain_grp.shift(1)
    df["fc3_rain_roll3"] = rain_grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["fc3_pest_cumsum"] = grp.transform(lambda x: x.shift(1).cumsum())

    for c in FC3:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_fc4(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    pest_norm = _pest_norm_feat(df, stats)
    df["fc4_cwpp"] = df["fc1_wbi"].clip(lower=0) / (df["fc1_pet"] * 12 + 1)

    # Legacy code name `tgsl`; scientifically interpreted as a normalized
    # thermal suitability proxy, not literal growing-season days.
    df["fc4_tgsl"] = (df["fc1_gdd"] / stats["thermal_proxy_max"]).clip(0, 1)

    rain_score = (1 - np.abs(df["rainfall"] - 900) / 900).clip(0, 1)
    temp_score = (1 - np.abs(df["temp_mean"] - 20) / 20).clip(0, 1)
    df["fc4_yps"] = (rain_score + temp_score + pest_norm) / 3
    water_def = np.abs(df["fc1_wbi"].clip(upper=0)) / 1000
    pest_vol = df.get("fc3_pest_roll3_std", pd.Series(0, index=df.index)) / (stats["pest_std"] + 1e-8)
    df["fc4_ari"] = 0.5 * water_def.clip(0, 1) + 0.5 * pest_vol.clip(0, 1)
    return df


def build_fc5(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    yn = _year_norm(df, stats)
    df["fc5_log_pest"] = np.log1p(df["pesticide_use"].clip(lower=0))
    df["fc5_log_rain"] = np.log1p(df["rainfall"].clip(lower=0))
    df["fc5_sqrt_gdd"] = np.sqrt(df["fc1_gdd"].clip(lower=0))
    df["fc5_rain_sq"] = df["rainfall"] ** 2 / 1e6
    df["fc5_temp_sq"] = df["temp_mean"] ** 2 / 100
    df["fc5_gdd_rain"] = df["fc1_gdd"] * df["rainfall"] / 1e5
    df["fc5_pest_year"] = df["fc5_log_pest"] * yn
    return df


def build_all_features(df: pd.DataFrame, train_mask: np.ndarray):
    df = build_fc1(df)
    stats = fit_feature_stats(df, train_mask)
    df = build_fc2(df, stats)
    df = build_fc3(df, stats)
    df = build_fc4(df, stats)
    df = build_fc5(df, stats)

    engineered = FC1 + FC2 + FC3 + FC4 + FC5
    df[engineered] = df[engineered].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"[features] Engineered features: {len(engineered)} (8+6+12+4+7)")
    return df, stats
