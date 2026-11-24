# src/config.py
# ============================================================
# KONFIGURASI TERPUSAT – Leakage-Controlled Revision
# ============================================================

import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH       = os.path.join(BASE_DIR, "yield_df.csv")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
DEPLOYMENT_DIR  = os.path.join(BASE_DIR, "deployment")

for _d in (OUTPUTS_DIR, DEPLOYMENT_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:  # filesystem read-only di beberapa hosting
        pass

SEED = 42
GROUP_COL = "Area"
TIME_COL = "Year"
ITEM_COL = "Item"
TARGET = "crop_yield"

TEST_YEAR_FRAC = 0.20
HOLDOUT_COUNTRY_FRAC = 0.20
N_SPLITS_GROUP = 5
N_SPLITS_RANDOM = 5
MIN_YEARS_PER_AREA = 15

# Raw/primitive predictors yang benar-benar masuk ke S1.
# Pesticide use wajib dimasukkan karena seluruh engineered interaction
# dibangun darinya dan manuscript mendefinisikannya sebagai primitive input.
PRIMITIF_BASE = [
    "rainfall",
    "temp_mean",
    "pesticide_use",
    "co2_ppm",
    "Year",
]
BASELINE_EXTRA_COLS = ["Year"]

FC1 = [
    "fc1_gdd", "fc1_ombrothermic", "fc1_lang_rf", "fc1_htc",
    "fc1_aridity_dm", "fc1_pet", "fc1_wbi", "fc1_wsr",
]
FC2 = [
    "fc2_pce", "fc2_pwi", "fc2_ocpi", "fc2_pipu", "fc2_tpe", "fc2_cfwi",
]
FC3 = [
    "fc3_pest_lag1", "fc3_pest_lag2", "fc3_pest_roll3", "fc3_pest_roll3_std",
    "fc3_pest_delta", "fc3_pest_pct_change", "fc3_year_sin", "fc3_year_cos",
    "fc3_decade_norm", "fc3_rain_lag1", "fc3_rain_roll3", "fc3_pest_cumsum",
]
FC4 = ["fc4_cwpp", "fc4_tgsl", "fc4_yps", "fc4_ari"]
FC5 = [
    "fc5_log_pest", "fc5_log_rain", "fc5_sqrt_gdd", "fc5_rain_sq",
    "fc5_temp_sq", "fc5_gdd_rain", "fc5_pest_year",
]
ALL_FC = FC1 + FC2 + FC3 + FC4 + FC5
FC_COUNTS = {"FC1": len(FC1), "FC2": len(FC2), "FC3": len(FC3), "FC4": len(FC4), "FC5": len(FC5)}


def build_scenarios(country_onehot_cols: list, item_onehot_cols: list):
    """Bangun cumulative scenarios dan single-module ablations."""
    primitif = PRIMITIF_BASE + country_onehot_cols + item_onehot_cols
    baseline = country_onehot_cols + item_onehot_cols + BASELINE_EXTRA_COLS
    all_feat = primitif + ALL_FC

    scenarios = {
        "S0 - Baseline (Country+Crop+Year)": baseline,
        "S1 - Primitif": primitif,
        "S2 - FC1+FC2": primitif + FC1 + FC2,
        "S3 - FC1+FC2+FC3": primitif + FC1 + FC2 + FC3,
        "S4 - Full FC": all_feat,
    }
    ablation_scenarios = {
        "A1 - Primitif+FC1": primitif + FC1,
        "A2 - Primitif+FC2": primitif + FC2,
        "A3 - Primitif+FC3": primitif + FC3,
        "A4 - Primitif+FC4": primitif + FC4,
        "A5 - Primitif+FC5": primitif + FC5,
    }
    return scenarios, ablation_scenarios, primitif, all_feat


# Annual Mauna Loa CO2 series. In interpretation, treat this variable as
# a temporal trend proxy, not as causal evidence of fertilization.
CO2_MAUNA_LOA = {
    1990: 354.4, 1991: 355.6, 1992: 356.4, 1993: 357.1, 1994: 358.9,
    1995: 360.9, 1996: 362.7, 1997: 363.8, 1998: 366.6, 1999: 368.4,
    2000: 369.6, 2001: 371.1, 2002: 373.2, 2003: 375.8, 2004: 377.5,
    2005: 379.8, 2006: 381.9, 2007: 383.8, 2008: 385.6, 2009: 387.4,
    2010: 389.9, 2011: 391.6, 2012: 393.9, 2013: 396.5, 2014: 398.6,
    2015: 400.8, 2016: 404.2,
}

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
