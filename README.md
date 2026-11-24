# Crop Yield Predictor (Streamlit)

Model final: LightGBM, skenario S3 (FC1+FC2+FC3).

## Struktur repo
```
app.py
requirements.txt
yield_df.csv                         <- dataset asli (dipakai untuk riwayat FC3)
src/                                 <- modul feature engineering dari notebook
deployment/
  best_model.pkl
  country_onehot_encoder.pkl
  item_onehot_encoder.pkl
  feature_stats.json
  metadata.json
  (opsional untuk tab Evaluasi) hasil_evaluasi_model.csv,
  heldout_country_validation_S1_S3.csv, hasil_crossvalidation.csv,
  ablation_per_module.csv, feature_importance_lightgbm.csv
```

## Langkah
1. Di Colab, jalankan pipeline sampai selesai lalu jalankan sel ZIP (bagian 7) dan unduh `skripsi_project_deployment.zip`.
2. Ekstrak isinya ke folder `deployment/` di proyek ini. Taruh juga `yield_df.csv` di root.
3. Tes lokal: `pip install -r requirements.txt` lalu `streamlit run app.py`.
4. Push ke GitHub, lalu buka https://share.streamlit.io, pilih repo, main file `app.py`, Python 3.11, Deploy.

Catatan: versi paket di requirements.txt disamakan dengan Colab (LightGBM 4.6.0, scikit-learn 1.6.1)
agar file .pkl terbaca tanpa masalah. Jangan ubah versi tanpa melatih ulang.
