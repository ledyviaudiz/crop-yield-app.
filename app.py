import json
import os

import pandas as pd
import streamlit as st

from src.config import DEPLOYMENT_DIR
from src.predict import (
    HIST_COLS, build_input_row, load_artifacts, load_history,
    model_feature_names, predict_yield,
)

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Memuat dataset...")
def get_history():
    return load_history()


def read_csv_if_exists(name):
    p = os.path.join(DEPLOYMENT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ------------------------------------------------------------------ load
try:
    model, country_enc, item_enc, stats, meta = get_artifacts()
    hist_all = get_history()
except FileNotFoundError as e:
    st.error(
        f"File tidak ditemukan: `{e.filename}`. Pastikan folder `deployment/` "
        "(best_model.pkl, country_onehot_encoder.pkl, item_onehot_encoder.pkl, "
        "feature_stats.json) dan `yield_df.csv` ada di root repo."
    )
    st.stop()

HAS_DATA = not hist_all.empty
if HAS_DATA:
    countries = sorted(hist_all["Area"].unique())
else:
    _hm = os.path.join(DEPLOYMENT_DIR, "heldout_country_manifest.json")
    _held = json.load(open(_hm, encoding="utf-8")).get("heldout_countries", []) if os.path.exists(_hm) else []
    countries = sorted(set(country_enc.categories_[0]) | set(_held))
    st.warning("`yield_df.csv` tidak ditemukan: nilai default dan riwayat tidak terisi otomatis, isi manual.")
crops = list(item_enc.categories_[0])
known_countries = set(country_enc.categories_[0])
n_features = len(model_feature_names(model))

# ------------------------------------------------------------------ header
st.title("🌾 Prediksi Hasil Panen Tingkat Negara")
st.caption(
    f"{meta.get('final_timebased_model', 'LightGBM')} · skenario "
    f"{meta.get('final_timebased_scenario', 'S3 - FC1+FC2+FC3')} · {n_features} fitur"
)

tab_pred, tab_eval, tab_about = st.tabs(["Prediksi", "Evaluasi model", "Tentang & batasan"])

# ------------------------------------------------------------------ prediksi
with tab_pred:
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Negara", countries, index=countries.index("Indonesia") if "Indonesia" in countries else 0)
    item = c2.selectbox("Tanaman", crops)
    year = c3.number_input("Tahun target", min_value=1990, max_value=2035, value=2014, step=1)
    year = int(year)

    combo = hist_all[(hist_all["Area"] == area) & (hist_all["Item"] == item)]
    if combo.empty:
        st.info("Kombinasi negara–tanaman ini tidak ada di dataset; isi input dan riwayat secara manual.")

    # nilai default: baris tahun target bila ada, kalau tidak baris terakhir sebelum tahun target
    prior = combo[combo["Year"] < year]
    exact = combo[combo["Year"] == year]
    base = exact.iloc[-1] if not exact.empty else (prior.iloc[-1] if not prior.empty else None)
    d_rain = float(base["rainfall"]) if base is not None else 1000.0
    d_temp = float(base["temp_mean"]) if base is not None else 20.0
    d_pest = float(base["pesticide_use"]) if base is not None else 100.0

    k = f"{area}|{item}|{year}"
    st.subheader("Input tahun target")
    i1, i2, i3 = st.columns(3)
    rain = i1.number_input("Curah hujan (mm/tahun)", 0.0, 10000.0, d_rain, 10.0, key=f"r|{k}")
    temp = i2.number_input("Suhu rata-rata (°C)", -10.0, 45.0, d_temp, 0.1, key=f"t|{k}")
    pest = i3.number_input("Pestisida (ton)", 0.0, 1_000_000.0, d_pest, 10.0, key=f"p|{k}")

    st.subheader("Riwayat tahun sebelumnya")
    st.caption(
        "Fitur FC3 (lag, rolling, kumulatif pestisida/hujan) dihitung dari riwayat ini. "
        "Default: 3 tahun terakhir dari dataset. Boleh diedit/ditambah; baris dengan tahun ≥ tahun target diabaikan."
    )
    default_hist = prior.tail(3)[HIST_COLS].reset_index(drop=True)
    hist_df = st.data_editor(
        default_hist, num_rows="dynamic", key=f"h|{k}",
        column_config={"Year": st.column_config.NumberColumn("Year", format="%d", step=1)},
    )

    if st.button("Prediksi", type="primary"):
        ignored = (hist_df["Year"] >= year).sum() if len(hist_df) else 0
        if ignored:
            st.warning(f"{ignored} baris riwayat diabaikan karena tahunnya ≥ tahun target.")
        row, c_known, i_known = build_input_row(
            area, item, year, rain, temp, pest, hist_df, stats, country_enc, item_enc
        )
        yhat = predict_yield(model, row)
        yhat = max(yhat, 0.0)

        m1, m2 = st.columns(2)
        m1.metric("Prediksi hasil panen", f"{yhat / 10000:,.2f} ton/ha")
        m2.metric("Dalam satuan dataset", f"{yhat:,.0f} hg/ha")

        if not c_known:
            st.warning(
                "Negara ini tidak ada di data training (termasuk negara held-out), sehingga "
                "indikator negara bernilai nol semua. Prediksi hanya bertumpu pada iklim, "
                "pestisida, dan tanaman, dan lebih tidak pasti."
            )
        if len(hist_df.query("Year < @year")) == 0:
            st.info("Tanpa riwayat, fitur FC3 bernilai 0 (sama seperti baris pertama tiap seri saat training).")
        if not exact.empty:
            actual = float(exact.iloc[-1]["crop_yield"])
            st.success(f"Nilai aktual di dataset ({year}): {actual / 10000:,.2f} ton/ha ({actual:,.0f} hg/ha)")

        with st.expander("Lihat fitur yang masuk ke model"):
            feats = model_feature_names(model)
            show = row.reindex(columns=feats, fill_value=0.0).T
            show.columns = ["nilai"]
            st.dataframe(show[~show.index.str.startswith(("country_", "crop_"))])

# ------------------------------------------------------------------ evaluasi
with tab_eval:
    if meta:
        a, b, c, d = st.columns(4)
        a.metric("R² (time-based test)", f"{meta.get('best_timebased_R2', float('nan')):.4f}")
        b.metric("RMSE (hg/ha)", f"{meta.get('best_timebased_RMSE', float('nan')):,.0f}")
        c.metric("Negara development / held-out",
                 f"{meta.get('countries_development', '?')} / {meta.get('countries_heldout', '?')}")
        d.metric("Split waktu", f"≤ {meta.get('time_split_threshold_year', '?')}")

    for title, fname in [
        ("Evaluasi semua skenario", "hasil_evaluasi_model.csv"),
        ("Validasi negara held-out (S1 vs S3)", "heldout_country_validation_S1_S3.csv"),
        ("Cross-validation", "hasil_crossvalidation.csv"),
        ("Ablasi per modul", "ablation_per_module.csv"),
    ]:
        dfc = read_csv_if_exists(fname)
        if dfc is not None:
            st.subheader(title)
            st.dataframe(dfc)

    imp = read_csv_if_exists("feature_importance_lightgbm.csv")
    if imp is not None:
        st.subheader("Feature importance (LightGBM)")
        top = imp[~imp["Feature"].str.startswith(("country_", "crop_"))].head(15)
        st.bar_chart(top.set_index("Feature")["Importance"])

# ------------------------------------------------------------------ tentang
with tab_about:
    st.markdown(
        """
**Pipeline.** Fitur dibangun dengan modul yang sama seperti saat training (FC1 indeks iklim,
FC2 interaksi pestisida–iklim, FC3 lag/rolling temporal). Statistik normalisasi diambil dari
`feature_stats.json` (di-fit hanya pada data training development).

**Batasan yang perlu dipahami pengguna**
- Model dilatih pada data tingkat **negara per tahun**, bukan lahan atau petani individu.
- `fc1_gdd` dan `fc1_pet` adalah *proxy* tahunan sederhana, bukan GDD harian atau Thornthwaite penuh.
- `co2_ppm` adalah proxy tren waktu (seri Mauna Loa sampai 2016), bukan bukti efek pemupukan CO₂.
  Untuk tahun > 2016 nilainya memakai nilai terdekat (2016).
- Untuk negara yang tidak ada di data training, indikator negara bernilai nol dan akurasi turun
  (lihat tab *Evaluasi model* bagian held-out).
- Hasil adalah estimasi statistik, bukan jaminan hasil panen.
"""
    )
