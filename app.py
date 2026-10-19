# =====================================================================
# Aplikasi Estimasi & Cek Kewajaran Harga Rumah Jabodetabek
# Proyek Akhir Pembelajaran Mesin
#
# Alur aplikasi:
#   Input spesifikasi -> Preprocessing (di dalam pipeline) -> Model
#   -> Estimasi harga -> (opsional) Cek kewajaran harga penawaran
#
# File yang dibutuhkan di folder yang sama:
#   model_terbaik.joblib  : pipeline preprocessing + model (dari notebook 03)
#   info_model.json       : nama model & metrik (dari notebook 03)
#   lokasi_district.csv   : tabel kecamatan -> koordinat (dari notebook 02)
# =====================================================================

import json                      # membaca info_model.json

import joblib                    # memuat model yang sudah dilatih
import numpy as np               # np.exp untuk mengembalikan log -> Rupiah
import pandas as pd              # menyusun input menjadi tabel
import streamlit as st           # membuat tampilan web

# ---------------------------------------------------------------------
# Pengaturan halaman
# ---------------------------------------------------------------------
st.set_page_config(page_title="Cek Harga Rumah Jabodetabek", page_icon="🏠", layout="centered")


# ---------------------------------------------------------------------
# Memuat model & data pendukung (disimpan di cache agar tidak dimuat ulang tiap klik)
# ---------------------------------------------------------------------
@st.cache_resource
def muat_model():
    return joblib.load("model_terbaik.joblib")          # pipeline: preprocessing + Gradient Boosting


@st.cache_data
def muat_info():
    with open("info_model.json") as f:
        return json.load(f)                              # nama model, MAPE, MAE, daftar fitur


@st.cache_data
def muat_lokasi():
    return pd.read_csv("lokasi_district.csv")            # kolom: city, district, lat, long


model = muat_model()
info = muat_info()
lokasi = muat_lokasi()

MAPE = info["mape_test_persen"] / 100                    # contoh: 22.95% -> 0.2295, dipakai sebagai batas toleransi


# ---------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------
def format_rp(x):
    """Format angka ke Rupiah singkat: Rp1,81 miliar / Rp850 juta."""
    if x >= 1e9:
        return "Rp" + f"{x / 1e9:.2f}".replace(".", ",") + " miliar"
    return "Rp" + f"{x / 1e6:,.0f}".replace(",", ".") + " juta"


def format_rp_lengkap(x):
    """Format angka ke Rupiah lengkap: Rp1.805.698.325."""
    return "Rp" + f"{x:,.0f}".replace(",", ".")


# Pilihan isian (sama dengan kategori yang dikenal model saat training)
PILIHAN_LISTRIK = [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 11000, 13200, 16500, 23000]
PILIHAN_SERTIFIKAT = {"SHM (Sertifikat Hak Milik)": "SHM", "HGB (Hak Guna Bangunan)": "HGB",
                      "Lainnya (PPJB, girik, adat, dll.)": "Lainnya", "Tidak tahu": np.nan}
PILIHAN_KONDISI = {"Baru": "baru", "Bagus sekali": "bagus sekali", "Bagus": "bagus",
                   "Sudah renovasi": "sudah renovasi", "Butuh renovasi": "butuh renovasi", "Tidak tahu": np.nan}
PILIHAN_PERABOT = {"Unfurnished (kosong)": "unfurnished", "Semi furnished": "semi furnished",
                   "Furnished (lengkap)": "furnished", "Tidak tahu": np.nan}


# ---------------------------------------------------------------------
# Judul
# ---------------------------------------------------------------------
st.title("🏠 Cek Harga Rumah Jabodetabek")
st.write(
    "Masukkan spesifikasi rumah untuk mendapatkan **estimasi harga pasar**. "
    "Jika ada harga yang ditawarkan penjual, aplikasi akan menilai apakah harganya **wajar**."
)

# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------
st.subheader("1. Lokasi")
kol1, kol2 = st.columns(2)
with kol1:
    kota = st.selectbox("Kota/Kabupaten", sorted(lokasi["city"].unique()))
with kol2:
    daftar_kecamatan = sorted(lokasi.loc[lokasi["city"] == kota, "district"].unique())   # kecamatan di kota terpilih
    kecamatan = st.selectbox("Kecamatan/Area", daftar_kecamatan)

titik = lokasi[(lokasi["city"] == kota) & (lokasi["district"] == kecamatan)].iloc[0]   # ambil lat/long kecamatan

with st.form("form_rumah"):
    st.subheader("2. Spesifikasi Rumah")
    k1, k2 = st.columns(2)
    with k1:
        luas_tanah = st.number_input("Luas tanah (m²)", min_value=15, max_value=10000, value=120, step=5)
        kamar_tidur = st.number_input("Kamar tidur", min_value=1, max_value=10, value=3)
        lantai = st.number_input("Jumlah lantai", min_value=1, max_value=5, value=2)
        carport = st.number_input("Carport", min_value=0, max_value=10, value=1)
    with k2:
        luas_bangunan = st.number_input("Luas bangunan (m²)", min_value=20, max_value=10000, value=100, step=5)
        kamar_mandi = st.number_input("Kamar mandi", min_value=1, max_value=10, value=2)
        listrik = st.selectbox("Daya listrik (VA)", PILIHAN_LISTRIK, index=PILIHAN_LISTRIK.index(2200))
        garasi = st.number_input("Garasi", min_value=0, max_value=10, value=0)

    k3, k4, k5 = st.columns(3)
    with k3:
        sertifikat = st.selectbox("Sertifikat", list(PILIHAN_SERTIFIKAT))
    with k4:
        kondisi = st.selectbox("Kondisi bangunan", list(PILIHAN_KONDISI), index=2)
    with k5:
        perabot = st.selectbox("Perabot", list(PILIHAN_PERABOT))

    st.subheader("3. Harga Penawaran (opsional)")
    penawaran_juta = st.number_input(
        "Harga yang ditawarkan penjual (dalam juta Rp). Isi 0 jika hanya ingin estimasi.",
        min_value=0, max_value=1_000_000, value=0, step=50,
        help="Contoh: 1500 = Rp1,5 miliar",
    )

    tombol = st.form_submit_button("Hitung Estimasi", type="primary", use_container_width=True)

# ---------------------------------------------------------------------
# PROSES & OUTPUT
# ---------------------------------------------------------------------
if tombol:
    # 1) Susun input menjadi 1 baris tabel dengan nama kolom yang sama seperti saat training
    data_input = pd.DataFrame([{
        "land_size_m2": luas_tanah,
        "building_size_m2": luas_bangunan,
        "bedrooms": kamar_tidur,
        "bathrooms": kamar_mandi,
        "carports": carport,
        "garages": garasi,
        "floors": lantai,
        "electricity": listrik,
        "lat": titik["lat"],
        "long": titik["long"],
        "city": kota,
        "certificate": PILIHAN_SERTIFIKAT[sertifikat],          # NaN -> diisi "Tidak diketahui" oleh pipeline
        "property_condition": PILIHAN_KONDISI[kondisi],
        "furnishing": PILIHAN_PERABOT[perabot],
    }])[info["fitur"]]                                         # urutkan kolom persis seperti saat training
    for kolom in ["city", "certificate", "property_condition", "furnishing"]:
        data_input[kolom] = data_input[kolom].astype(object)   # pastikan kolom kategori bertipe teks, walau isinya NaN

    # 2) Prediksi: model menghasilkan log(harga), dikembalikan ke Rupiah dengan exp
    estimasi = float(np.exp(model.predict(data_input)[0]))
    batas_bawah = estimasi * (1 - MAPE)
    batas_atas = estimasi * (1 + MAPE)

    st.divider()
    st.subheader("Hasil Estimasi")
    st.metric("Estimasi harga pasar", format_rp(estimasi))
    st.write(
        f"Kisaran wajar: **{format_rp(batas_bawah)} – {format_rp(batas_atas)}** "
        f"(± {info['mape_test_persen']:.1f}%, sesuai rata-rata kesalahan model)"
    )
    st.caption(
        f"Setara {format_rp_lengkap(estimasi / luas_bangunan)} per m² bangunan · "
        f"Lokasi: {kecamatan}, {kota}"
    )

    # 3) Peringatan jika input di luar jangkauan data latih
    if luas_bangunan > 1500 or luas_tanah > 3000:
        st.warning("Ukuran rumah ini jauh di atas sebagian besar data latih, jadi estimasi kurang akurat.")
    if luas_bangunan > luas_tanah * lantai:
        st.warning("Luas bangunan lebih besar dari luas tanah × jumlah lantai. Periksa kembali isian.")

    # 4) Cek kewajaran harga penawaran
    if penawaran_juta > 0:
        penawaran = penawaran_juta * 1e6
        selisih = (penawaran - estimasi) / estimasi * 100          # + berarti lebih mahal dari estimasi

        st.subheader("Cek Kewajaran Harga")
        k1, k2 = st.columns(2)
        k1.metric("Harga penawaran", format_rp(penawaran))
        k2.metric("Selisih dari estimasi", f"{selisih:+.1f}%")

        if abs(selisih) <= info["mape_test_persen"]:
            st.success(
                f"✅ **WAJAR**: harga penawaran masih dalam kisaran estimasi pasar "
                f"(selisih {selisih:+.1f}%, toleransi ±{info['mape_test_persen']:.1f}%)."
            )
        elif selisih > 0:
            st.error(
                f"⬆️ **DI ATAS PASAR**: harga penawaran {selisih:.1f}% lebih mahal dari estimasi. "
                "Ada ruang untuk negosiasi, kecuali rumah punya keunggulan yang tidak tercatat di data "
                "(misalnya fasilitas premium atau lokasi pinggir jalan utama)."
            )
        else:
            st.warning(
                f"⬇️ **DI BAWAH PASAR**: harga penawaran {abs(selisih):.1f}% lebih murah dari estimasi. "
                "Bisa jadi kesempatan bagus, tapi periksa dulu legalitas sertifikat dan kondisi fisik rumah."
            )

# ---------------------------------------------------------------------
# Informasi model
# ---------------------------------------------------------------------
with st.expander("ℹ️ Tentang model"):
    st.write(f"**Model:** {info['nama_model']}")
    st.write(f"**Rata-rata kesalahan (MAPE) di data uji:** {info['mape_test_persen']:.2f}%")
    st.write(f"**R² (skala log) di data uji:** {info['r2_log_test']:.3f}")
    st.write(f"**Jumlah data latih:** {info['jumlah_data_train']:,} rumah".replace(",", "."))
    st.write(
        "**Sumber data:** Daftar Harga Rumah Jabodetabek (Kaggle), hasil scraping iklan rumah123.com. "
        "Harga di data adalah **harga penawaran iklan**, bukan harga transaksi akhir."
    )
    st.write(
        "**Batas toleransi kewajaran** memakai MAPE model, sehingga status \"wajar\" "
        "memperhitungkan ketidakpastian prediksi."
    )

st.caption("Proyek Akhir Pembelajaran Mesin · Estimasi ini bukan penilaian (appraisal) resmi.")
