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
import os                        # mengecek apakah file model pembanding ada

import joblib                    # memuat model yang sudah dilatih
import numpy as np               # np.exp untuk mengembalikan log -> Rupiah
import pandas as pd              # menyusun input menjadi tabel
import streamlit as st           # membuat tampilan web

# =====================================================================
# PENGATURAN MODEL  ->  model utama yang dipakai versi aplikasi ini
# =====================================================================
FILE_MODEL_UTAMA = "model_xgboost.joblib"   # file model utama
FILE_INFO = "info_model_xgboost.json"   # file info/metrik model utama
LABEL_VERSI = "Versi XGBoost"           # label kecil di bagian atas halaman

# Semua model yang bisa ditampilkan di bagian perbandingan (file yang belum ada akan dilewati)
FILE_SEMUA_MODEL = {
    "Linear Regression": "model_linear_regression.joblib",
    "Random Forest": "model_random_forest.joblib",
    "Gradient Boosting": "model_terbaik.joblib",
    "XGBoost": "model_xgboost.joblib",
}
# File info (metrik) tiap model, dipakai untuk menampilkan MAPE di kartu & tabel perbandingan
FILE_SEMUA_INFO = {
    "Linear Regression": "info_model_linear_regression.json",
    "Random Forest": "info_model_random_forest.json",
    "Gradient Boosting": "info_model.json",
    "XGBoost": "info_model_xgboost.json",
}

# =====================================================================
# PENGATURAN TAMPILAN  ->  ubah nilai di bagian ini untuk mengganti look
# =====================================================================
NAMA_BRAND = "RUMAHWAJAR"                         # nama di pojok kiri atas
FONT = "Manrope"                                  # nama font dari fonts.google.com
WARNA_LATAR = "#ECECEA"                           # latar halaman (abu muda)
WARNA_KARTU = "#FFFFFF"                           # latar kartu putih
WARNA_TEKS = "#141414"                            # teks utama (hampir hitam)
WARNA_TEKS_REDUP = "#6B6B68"                      # teks keterangan (abu)
WARNA_AKSEN = "#141414"                           # tombol & pil aktif (hitam)
WARNA_GARIS = "#DEDEDA"                           # garis tipis
SUDUT = "24px"                                    # kelengkungan sudut kartu
JUDUL_HERO = "Temukan harga wajar<br>rumah impianmu"
DESKRIPSI_HERO = (
    "Masukkan spesifikasi rumah di Jabodetabek dan dapatkan estimasi harga pasar "
    "dari model Machine Learning. Punya harga dari penjual? Kami bantu menilai "
    "apakah harganya wajar, terlalu mahal, atau terlalu murah."
)
LABEL_HERO = "Estimasi · Bandingkan · Putuskan"
# =====================================================================

st.set_page_config(page_title="Cek Harga Rumah Jabodetabek", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------
# CSS: semua gaya tampilan ada di sini (memakai pengaturan di atas)
# ---------------------------------------------------------------------
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family={FONT.replace(" ", "+")}:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {{ font-family: '{FONT}', sans-serif !important; }}
.stApp {{ background: {WARNA_LATAR}; color: {WARNA_TEKS}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ color: {WARNA_TEKS}; letter-spacing: -0.03em; }}
/* navbar */
.nav {{ display:flex; align-items:center; justify-content:space-between; padding:14px 8px 6px; }}
.brand {{ font-weight:800; font-size:18px; letter-spacing:0.02em; display:flex; align-items:center; gap:8px; color:{WARNA_TEKS}; }}
.brand-dot {{ width:18px; height:18px; border-radius:50%; background:{WARNA_AKSEN}; display:inline-block; box-shadow: 6px 0 0 -2px {WARNA_LATAR}, 6px 0 0 0 {WARNA_AKSEN}; }}
.menu {{ display:flex; gap:4px; background:{WARNA_AKSEN}; padding:5px; border-radius:999px; }}
.menu a {{ color:#FFFFFF; text-decoration:none; font-size:13px; padding:7px 16px; border-radius:999px; }}
.menu a.aktif {{ background:#FFFFFF; color:{WARNA_TEKS}; font-weight:600; }}
.nav-kanan {{ display:flex; gap:14px; align-items:center; font-size:13px; }}
.pil-putih {{ background:#FFFFFF; padding:8px 16px; border-radius:999px; font-weight:600; color:{WARNA_TEKS}; text-decoration:none; }}
/* hero */
.hero {{ background:{WARNA_KARTU}; border-radius:{SUDUT}; padding:44px 48px 0; margin-top:14px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.hero-atas {{ display:flex; justify-content:space-between; gap:48px; flex-wrap:wrap; }}
.chip {{ display:inline-flex; align-items:center; gap:8px; font-size:12px; padding:6px 12px; border-radius:999px; background:#F3F3F1; color:{WARNA_TEKS}; }}
.chip::before {{ content:''; width:7px; height:7px; border-radius:50%; background:{WARNA_AKSEN}; }}
.judul-hero {{ font-size:54px; line-height:1.04; font-weight:700; letter-spacing:-0.04em; margin:18px 0 0; color:{WARNA_TEKS}; }}
.desk-hero {{ max-width:420px; font-size:15px; line-height:1.6; color:{WARNA_TEKS_REDUP}; margin-top:56px; }}
.tombol-hitam {{ display:inline-flex; align-items:center; gap:8px; background:{WARNA_AKSEN}; color:#FFFFFF !important; text-decoration:none; padding:10px 18px; border-radius:999px; font-size:14px; font-weight:600; margin-top:18px; }}
.ilustrasi {{ display:flex; justify-content:center; margin-top:12px; }}
/* judul bagian */
.bagian {{ display:flex; justify-content:space-between; align-items:flex-end; gap:40px; flex-wrap:wrap; margin:48px 4px 18px; }}
.judul-bagian {{ font-size:40px; line-height:1.08; font-weight:700; letter-spacing:-0.035em; margin:12px 0 0; color:{WARNA_TEKS}; }}
.desk-bagian {{ max-width:430px; font-size:14px; line-height:1.6; color:{WARNA_TEKS_REDUP}; }}
/* kartu form (st.container dengan key="kartu_form") */
.st-key-kartu_form {{ background:{WARNA_KARTU}; border-radius:{SUDUT}; padding:28px 30px 18px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.st-key-kartu_form [data-testid="stForm"] {{ border:none; padding:0; }}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{ border-radius:14px !important; background:#F5F5F3 !important; border-color:transparent !important; }}
[data-testid="stFormSubmitButton"] button, .stButton button {{ border-radius:999px !important; padding:10px 22px !important; font-weight:600 !important; }}
[data-testid="stWidgetLabel"] p {{ font-size:13px !important; color:{WARNA_TEKS_REDUP} !important; }}
/* kartu hasil */
.grid-hasil {{ display:grid; grid-template-columns: 1.3fr 1fr 1fr; gap:16px; margin-top:22px; }}
.kartu {{ background:{WARNA_KARTU}; border-radius:{SUDUT}; padding:24px 26px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.label-kecil {{ font-size:12px; color:{WARNA_TEKS_REDUP}; text-transform:uppercase; letter-spacing:.08em; }}
.harga-besar {{ font-size:38px; font-weight:800; letter-spacing:-0.03em; margin:8px 0 4px; color:{WARNA_TEKS}; }}
.teks-redup {{ font-size:13px; color:{WARNA_TEKS_REDUP}; line-height:1.55; }}
.pil-status {{ display:inline-block; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:700; margin:10px 0 6px; }}
.wajar {{ background:#E3F1E6; color:#1D5B2C; }}
.mahal {{ background:#F7E1DC; color:#8A2A16; }}
.murah {{ background:#F6EDCF; color:#6B4E00; }}
/* perbandingan model */
.grid-model {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-top:16px; }}
.kartu-dipakai {{ outline:2px solid {WARNA_AKSEN}; }}
.lencana {{ display:inline-block; font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px; background:{WARNA_AKSEN}; color:#FFFFFF; margin-left:8px; vertical-align:middle; }}
.judul-model {{ font-size:16px; font-weight:700; color:{WARNA_TEKS}; }}
@media (max-width: 800px) {{ .grid-model {{ grid-template-columns:1fr; }} }}
/* cari area */
.st-key-kartu_area {{ background:{WARNA_KARTU}; border-radius:{SUDUT}; padding:28px 30px 18px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.grid-ringkas {{ display:grid; grid-template-columns: 1.3fr 1fr 1fr; gap:16px; margin-top:22px; }}
.kartu-rek {{ display:grid; grid-template-columns: 44px 1.6fr 1fr 1fr 1.2fr; gap:14px; align-items:center; background:{WARNA_KARTU}; border-radius:18px; padding:16px 22px; margin-top:10px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.nomor {{ width:36px; height:36px; border-radius:50%; background:{WARNA_AKSEN}; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:800; }}
.nama-area {{ font-weight:700; font-size:16px; color:{WARNA_TEKS}; }}
.nilai {{ font-weight:700; font-size:15px; color:{WARNA_TEKS}; }}
.batang {{ height:8px; border-radius:999px; background:#EDEDEA; overflow:hidden; margin-top:6px; }}
.isi-batang {{ height:100%; background:{WARNA_AKSEN}; border-radius:999px; }}
.premium {{ background:#F7E1DC; color:#8A2A16; }}
.atas {{ background:#F6EDCF; color:#6B4E00; }}
.rata {{ background:#ECECEA; color:#141414; }}
.terjangkau {{ background:#E3F1E6; color:#1D5B2C; }}
@media (max-width: 800px) {{ .grid-ringkas {{ grid-template-columns:1fr; }} .kartu-rek {{ grid-template-columns: 44px 1fr 1fr; row-gap:12px; }} .rek-nama {{ grid-column: 2 / -1; }} .rek-jarak {{ grid-column: 2; }} .rek-skor {{ grid-column: 1 / -1; }} }}
/* strip angka */
.strip {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:48px; }}
.strip .kartu {{ flex:1; min-width:180px; }}
.angka {{ font-size:30px; font-weight:800; letter-spacing:-0.03em; color:{WARNA_TEKS}; }}
.footer {{ text-align:center; font-size:12px; color:{WARNA_TEKS_REDUP}; margin-top:40px; }}
@media (max-width: 800px) {{
  .judul-hero {{ font-size:38px; }} .desk-hero {{ margin-top:16px; }}
  .menu, .nav-kanan {{ display:none; }} .grid-hasil {{ grid-template-columns:1fr; }}
  .hero {{ padding:28px 22px 0; }}
}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Memuat model & data pendukung (disimpan di cache agar tidak dimuat ulang tiap klik)
# ---------------------------------------------------------------------
@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL_UTAMA)                 # pipeline: preprocessing + model utama


@st.cache_data
def muat_info():
    with open(FILE_INFO) as f:
        return json.load(f)                              # nama model, MAPE, MAE, daftar fitur


@st.cache_data
def muat_lokasi():
    return pd.read_csv("lokasi_district.csv")            # kolom: city, district, lat, long


# Model pembanding (opsional). Jika file-nya belum di-upload, aplikasi tetap jalan tanpa perbandingan.
@st.cache_resource
def muat_model_pembanding(nama_utama):
    hasil = {}
    for nama, file in FILE_SEMUA_MODEL.items():
        if nama != nama_utama and os.path.exists(file):  # lewati model utama & file yang belum ada
            try:
                hasil[nama] = joblib.load(file)
            except Exception:                            # mis. library model belum terpasang -> lewati saja
                pass
    return hasil


@st.cache_data
def muat_perbandingan():
    hasil = {}
    if os.path.exists("perbandingan_model.json"):        # metrik model dari notebook 03 (jika ada)
        with open("perbandingan_model.json") as f:
            hasil = json.load(f)
    kolom = ["mape_test_persen", "mae_test_rp", "r2_log_test", "r2_log_train"]
    for nama, file in FILE_SEMUA_INFO.items():           # lengkapi dari file info tiap versi
        if os.path.exists(file):
            with open(file) as f:
                isi = json.load(f)
            hasil[nama] = {k: isi[k] for k in kolom if k in isi}
    urutan = [n for n in FILE_SEMUA_MODEL if n in hasil] # urutkan: LR, RF, GB, XGBoost
    return {n: hasil[n] for n in urutan}


# Data hasil notebook 04 (opsional). Jika belum di-upload, bagian "Cari Area" disembunyikan.
@st.cache_data
def muat_data_area():
    file_area = ["harga_per_area.csv", "titik_acuan.csv", "ringkasan_titik_acuan.csv"]
    if not all(os.path.exists(f) for f in file_area):
        return None
    return tuple(pd.read_csv(f) for f in file_area)


model = muat_model()
info = muat_info()
lokasi = muat_lokasi()
model_pembanding = muat_model_pembanding(info["nama_model"])
perbandingan = muat_perbandingan()
data_area = muat_data_area()

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


def html(teks):
    """Tampilkan potongan HTML (tanpa baris kosong agar tidak dibaca sebagai teks biasa)."""
    st.markdown(teks, unsafe_allow_html=True)


# Gambar rumah sederhana (SVG buatan sendiri, warna abu-abu seperti contoh desain)
GAMBAR_RUMAH = (
    '<svg viewBox="0 0 760 300" width="100%" style="max-width:760px" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="380" cy="292" rx="350" ry="10" fill="#E6E6E3"/>'
    '<rect x="120" y="150" width="300" height="140" fill="#D9D9D6"/>'
    '<rect x="96" y="132" width="350" height="20" fill="#BDBDB9"/>'
    '<rect x="230" y="40" width="330" height="96" fill="#CFCFCB"/>'
    '<rect x="210" y="24" width="372" height="18" fill="#A9A9A5"/>'
    '<rect x="250" y="58" width="150" height="60" fill="#8F9396"/>'
    '<rect x="252" y="60" width="72" height="56" fill="#B9C2C7"/>'
    '<rect x="326" y="60" width="72" height="56" fill="#AEB8BE"/>'
    '<rect x="420" y="58" width="120" height="60" fill="#E9E9E6"/>'
    '<line x1="420" y1="88" x2="540" y2="88" stroke="#C9C9C5" stroke-width="2"/>'
    '<rect x="230" y="118" width="190" height="4" fill="#9E9E9A"/>'
    '<line x1="236" y1="100" x2="236" y2="122" stroke="#9E9E9A" stroke-width="2"/>'
    '<line x1="414" y1="100" x2="414" y2="122" stroke="#9E9E9A" stroke-width="2"/>'
    '<line x1="236" y1="100" x2="414" y2="100" stroke="#9E9E9A" stroke-width="2"/>'
    '<rect x="140" y="172" width="120" height="118" fill="#8F9396"/>'
    '<rect x="143" y="175" width="56" height="112" fill="#C7B39A"/>'
    '<rect x="201" y="175" width="56" height="112" fill="#BBA68C"/>'
    '<rect x="280" y="172" width="120" height="70" fill="#8F9396"/>'
    '<rect x="283" y="175" width="114" height="64" fill="#B9C2C7"/>'
    '<rect x="446" y="136" width="170" height="154" fill="#C4C4C0"/>'
    '<rect x="470" y="178" width="122" height="112" fill="#7D8184"/>'
    '<line x1="470" y1="200" x2="592" y2="200" stroke="#6E7275"/>'
    '<line x1="470" y1="222" x2="592" y2="222" stroke="#6E7275"/>'
    '<line x1="470" y1="244" x2="592" y2="244" stroke="#6E7275"/>'
    '<line x1="470" y1="266" x2="592" y2="266" stroke="#6E7275"/>'
    '<rect x="96" y="286" width="560" height="6" fill="#B5B5B1"/>'
    '<rect x="626" y="250" width="18" height="36" fill="#9C9C98"/>'
    '<circle cx="635" cy="236" r="20" fill="#7E8A80"/>'
    '<circle cx="622" cy="246" r="12" fill="#8E9A90"/>'
    '</svg>'
)

# Pilihan isian (sama dengan kategori yang dikenal model saat training)
PILIHAN_LISTRIK = [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 11000, 13200, 16500, 23000]
PILIHAN_SERTIFIKAT = {"SHM (Sertifikat Hak Milik)": "SHM", "HGB (Hak Guna Bangunan)": "HGB",
                      "Lainnya (PPJB, girik, adat, dll.)": "Lainnya", "Tidak tahu": np.nan}
PILIHAN_KONDISI = {"Baru": "baru", "Bagus sekali": "bagus sekali", "Bagus": "bagus",
                   "Sudah renovasi": "sudah renovasi", "Butuh renovasi": "butuh renovasi", "Tidak tahu": np.nan}
PILIHAN_PERABOT = {"Unfurnished (kosong)": "unfurnished", "Semi furnished": "semi furnished",
                   "Furnished (lengkap)": "furnished", "Tidak tahu": np.nan}

# ---------------------------------------------------------------------
# NAVBAR + HERO
# ---------------------------------------------------------------------
html(
    f'<div class="nav"><div class="brand"><span class="brand-dot"></span>&nbsp;{NAMA_BRAND}</div>'
    '<div class="menu"><a class="aktif" href="#beranda">Beranda</a><a href="#cek-harga">Cek Harga</a>'
    '<a href="#cari-area">Cari Area</a><a href="#tentang">Tentang Model</a></div>'
    '<div class="nav-kanan"><span>ID</span><a class="pil-putih" href="#cek-harga">Mulai</a></div></div>'
)
html(
    '<div class="hero" id="beranda"><div class="hero-atas">'
    f'<div><span class="chip">{LABEL_HERO}</span>&nbsp;<span class="chip">{LABEL_VERSI}</span>'
    f'<div class="judul-hero">{JUDUL_HERO}</div></div>'
    f'<div class="desk-hero">{DESKRIPSI_HERO}<br><a class="tombol-hitam" href="#cek-harga">Mulai cek harga &rarr;</a></div>'
    f'</div><div class="ilustrasi">{GAMBAR_RUMAH}</div></div>'
)

# ---------------------------------------------------------------------
# BAGIAN INPUT
# ---------------------------------------------------------------------
html(
    '<div class="bagian" id="cek-harga"><div><span class="chip">Cek Harga</span>'
    '<div class="judul-bagian">Cek harga<br>rumah incaranmu</div></div>'
    '<div class="desk-bagian">Pilih kota dan kecamatan, isi spesifikasi rumah, lalu (opsional) '
    'masukkan harga yang ditawarkan penjual untuk melihat apakah harganya wajar.</div></div>'
)

with st.container(key="kartu_form"):
    daftar_kota = sorted(lokasi["city"].unique())
    kota = st.pills("Kota/Kabupaten", daftar_kota, selection_mode="single", default=daftar_kota[0])
    if kota is None:                                        # jika pil dibatalkan, pakai kota pertama
        kota = daftar_kota[0]

    daftar_kecamatan = sorted(lokasi.loc[lokasi["city"] == kota, "district"].unique())   # kecamatan di kota terpilih
    kecamatan = st.selectbox("Kecamatan/Area", daftar_kecamatan)
    titik = lokasi[(lokasi["city"] == kota) & (lokasi["district"] == kecamatan)].iloc[0]  # ambil lat/long kecamatan

    with st.form("form_rumah"):
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            luas_tanah = st.number_input("Luas tanah (m²)", min_value=15, max_value=10000, value=120, step=5)
            carport = st.number_input("Carport", min_value=0, max_value=10, value=1)
        with k2:
            luas_bangunan = st.number_input("Luas bangunan (m²)", min_value=20, max_value=10000, value=100, step=5)
            garasi = st.number_input("Garasi", min_value=0, max_value=10, value=0)
        with k3:
            kamar_tidur = st.number_input("Kamar tidur", min_value=1, max_value=10, value=3)
            lantai = st.number_input("Jumlah lantai", min_value=1, max_value=5, value=2)
        with k4:
            kamar_mandi = st.number_input("Kamar mandi", min_value=1, max_value=10, value=2)
            listrik = st.selectbox("Daya listrik (VA)", PILIHAN_LISTRIK, index=PILIHAN_LISTRIK.index(2200))

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            sertifikat = st.selectbox("Sertifikat", list(PILIHAN_SERTIFIKAT))
        with k6:
            kondisi = st.selectbox("Kondisi bangunan", list(PILIHAN_KONDISI), index=2)
        with k7:
            perabot = st.selectbox("Perabot", list(PILIHAN_PERABOT))
        with k8:
            penawaran_juta = st.number_input(
                "Harga penawaran (juta Rp, opsional)", min_value=0, max_value=1_000_000, value=0, step=50,
                help="Contoh: 1500 = Rp1,5 miliar. Isi 0 jika hanya ingin estimasi.",
            )

        tombol = st.form_submit_button("Hitung estimasi  →", type="primary")

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

    # 3) Kartu status kewajaran (jika harga penawaran diisi)
    if penawaran_juta > 0:
        penawaran = penawaran_juta * 1e6
        selisih = (penawaran - estimasi) / estimasi * 100      # + berarti lebih mahal dari estimasi
        if abs(selisih) <= info["mape_test_persen"]:
            kelas, status, saran = "wajar", "WAJAR", "Harga masih dalam kisaran estimasi pasar."
        elif selisih > 0:
            kelas, status, saran = "mahal", "DI ATAS PASAR", "Ada ruang negosiasi, kecuali rumah punya keunggulan yang tidak tercatat di data."
        else:
            kelas, status, saran = "murah", "DI BAWAH PASAR", "Bisa jadi kesempatan bagus; periksa dulu legalitas dan kondisi fisik rumah."
        kartu_status = (
            f'<div class="kartu"><div class="label-kecil">Cek kewajaran</div>'
            f'<div class="pil-status {kelas}">{status}</div>'
            f'<div class="teks-redup">Penawaran {format_rp(penawaran)} · selisih '
            f'<b>{selisih:+.1f}%</b> (toleransi ±{info["mape_test_persen"]:.1f}%)'.replace(".", ",")
            + f'. {saran}</div></div>'
        )
    else:
        kartu_status = (
            '<div class="kartu"><div class="label-kecil">Cek kewajaran</div>'
            '<div class="teks-redup" style="margin-top:12px">Isi <b>harga penawaran</b> untuk melihat apakah '
            'harga dari penjual wajar, terlalu mahal, atau terlalu murah.</div></div>'
        )

    html(
        '<div class="grid-hasil">'
        f'<div class="kartu"><div class="label-kecil">Estimasi harga pasar</div>'
        f'<div class="harga-besar">{format_rp(estimasi)}</div>'
        f'<div class="teks-redup">Kisaran wajar {format_rp(batas_bawah)} – {format_rp(batas_atas)}<br>'
        f'{kecamatan}, {kota} · {luas_tanah}/{luas_bangunan} m² · {kamar_tidur} KT · {kamar_mandi} KM</div></div>'
        f'<div class="kartu"><div class="label-kecil">Harga per m² bangunan</div>'
        f'<div class="harga-besar" style="font-size:28px">{format_rp_lengkap(estimasi / luas_bangunan)}</div>'
        f'<div class="teks-redup">Estimasi dibagi luas bangunan</div></div>'
        f'{kartu_status}</div>'
    )

    # 4) Perbandingan semua model untuk rumah yang sama (hanya jika file model pembanding tersedia)
    if model_pembanding:
        nama_utama = info["nama_model"]
        semua_model = {n: (model if n == nama_utama else model_pembanding[n])   # urutan: LR, RF, GB, XGBoost
                       for n in FILE_SEMUA_MODEL if n == nama_utama or n in model_pembanding}
        kartu_model = ""
        for nama, m in semua_model.items():
            est = float(np.exp(m.predict(data_input)[0]))                  # estimasi model ini (Rupiah)
            beda = (est - estimasi) / estimasi * 100                        # selisih terhadap model utama
            mape_m = perbandingan.get(nama, {}).get("mape_test_persen")
            if nama == nama_utama:
                mape_m = info["mape_test_persen"]
            dipakai = nama == nama_utama
            teks_mape = f"MAPE data uji {mape_m:.2f}%".replace(".", ",") if mape_m else ""
            teks_beda = "Model utama versi ini" if dipakai else f"{beda:+.1f}% dari {nama_utama}".replace(".", ",")
            kartu_model += (
                f'<div class="kartu{" kartu-dipakai" if dipakai else ""}">'
                f'<div class="judul-model">{nama}{"<span class=lencana>DIPAKAI</span>" if dipakai else ""}</div>'
                f'<div class="harga-besar" style="font-size:30px">{format_rp(est)}</div>'
                f'<div class="teks-redup">{teks_mape}<br>{teks_beda}</div></div>'
            )
        html(
            '<div class="bagian" style="margin-top:36px"><div><span class="chip">Perbandingan Model</span>'
            f'<div class="judul-bagian" style="font-size:30px">Estimasi dari {len(semua_model)} model</div></div>'
            f'<div class="desk-bagian">Semua model diberi data rumah yang sama. Versi aplikasi ini memakai '
            f'<b>{nama_utama}</b> untuk estimasi utama dan cek kewajaran.</div></div>'
            f'<div class="grid-model">{kartu_model}</div>'
        )

    # 5) Peringatan jika input di luar jangkauan data latih
    if luas_bangunan > 1500 or luas_tanah > 3000:
        st.warning("Ukuran rumah ini jauh di atas sebagian besar data latih, jadi estimasi kurang akurat.")
    if luas_bangunan > luas_tanah * lantai:
        st.warning("Luas bangunan lebih besar dari luas tanah × jumlah lantai. Periksa kembali isian.")

# ---------------------------------------------------------------------
# CARI AREA STRATEGIS (dari notebook 04)
# ---------------------------------------------------------------------
def jarak_km(lat1, lon1, lat2, lon2):
    """Jarak garis lurus (rumus Haversine) dalam km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


KELAS_CSS = {"Premium": "premium", "Di atas rata-rata": "atas", "Rata-rata": "rata", "Terjangkau": "terjangkau"}
PRIORITAS = {"Lebih dekat": 0.7, "Seimbang": 0.5, "Lebih hemat": 0.3}   # bobot jarak pada skor

if data_area is not None:
    area, titik_acuan, ringkasan = data_area
    html(
        '<div class="bagian" id="cari-area"><div><span class="chip">Cari Area</span>'
        '<div class="judul-bagian">Cari area strategis<br>dekat kampus & industri</div></div>'
        '<div class="desk-bagian">Pilih kampus atau kawasan industri tujuanmu, masukkan budget, '
        'dan kami rekomendasikan area terdekat yang paling hemat. Harga memakai '
        '<b>rumah standar 70/90</b> (LB 70 m², LT 90 m², 2 KT, 1 KM) agar antar-area bisa dibandingkan adil.</div></div>'
    )

    with st.container(key="kartu_area"):
        jenis = st.pills("Saya mencari rumah dekat", ["Kampus", "Kawasan industri"],
                         selection_mode="single", default="Kampus", key="jenis_tujuan")
        kategori = "Industri" if jenis == "Kawasan industri" else "Kampus"
        a1, a2, a3, a4 = st.columns([2, 1, 1, 1])
        with a1:
            pilihan_tujuan = titik_acuan.loc[titik_acuan["kategori"] == kategori, "nama"].tolist()
            tujuan = st.selectbox("Tujuan", pilihan_tujuan, key="tujuan")
        with a2:
            budget_juta = st.number_input("Budget (juta Rp)", min_value=100, max_value=20_000,
                                          value=900, step=50, key="budget")
        with a3:
            jarak_maks = st.slider("Jarak maksimal (km)", 2, 25, 10, key="jarak_maks")
        with a4:
            prioritas = st.select_slider("Prioritas", list(PRIORITAS), value="Seimbang", key="prioritas")

    budget = budget_juta * 1e6
    t = titik_acuan[titik_acuan["nama"] == tujuan].iloc[0]

    # 1) Ringkasan harga di sekitar tujuan (radius 5 km, dari notebook 04)
    r = ringkasan[ringkasan["nama"] == tujuan]
    if not r.empty:
        r = r.iloc[0]
        indeks = int(round(r["indeks_harga"]))
        teks_iklan = format_rp(r["median_harga_iklan"]) if pd.notna(r["median_harga_iklan"]) else "-"
        html(
            '<div class="grid-ringkas">'
            f'<div class="kartu"><div class="label-kecil">Rumah 70/90 di sekitar {tujuan} (5 km)</div>'
            f'<div class="harga-besar">{format_rp(r["harga_standar_median"])}</div>'
            f'<div class="pil-status {KELAS_CSS.get(r["kelas_harga"], "rata")}">{r["kelas_harga"]}</div>'
            f'<div class="teks-redup">Indeks harga {indeks} (100 = rata-rata Jabodetabek)</div></div>'
            f'<div class="kartu"><div class="label-kecil">Median harga iklan sekitar</div>'
            f'<div class="harga-besar" style="font-size:28px">{teks_iklan}</div>'
            f'<div class="teks-redup">Semua ukuran rumah, dari {int(r["jumlah_iklan"])} iklan</div></div>'
            f'<div class="kartu"><div class="label-kecil">Rentang antar-area sekitar</div>'
            f'<div class="harga-besar" style="font-size:24px">{format_rp(r["harga_standar_min"])} – '
            f'{format_rp(r["harga_standar_max"])}</div>'
            f'<div class="teks-redup">Rumah 70/90 di {int(r["jumlah_area"])} area dalam radius 5 km</div></div>'
            '</div>'
        )

    # 2) Rekomendasi: area dalam radius, harga <= budget, data cukup -> urutkan dengan skor strategis
    hasil = area.copy()
    hasil["jarak_km"] = jarak_km(hasil["lat"].values, hasil["long"].values, t["lat"], t["long"])
    dalam_radius = hasil[hasil["jarak_km"] <= jarak_maks]
    cocok = dalam_radius[(dalam_radius["harga_standar"] <= budget) & dalam_radius["data_cukup"]].copy()
    bobot = PRIORITAS[prioritas]
    cocok["skor"] = 100 * (bobot * (1 - cocok["jarak_km"] / jarak_maks)
                           + (1 - bobot) * (1 - cocok["harga_standar"] / budget))
    cocok = cocok.sort_values("skor", ascending=False).head(5)

    html(f'<div class="bagian" style="margin-top:32px"><div><span class="chip">Rekomendasi</span>'
         f'<div class="judul-bagian" style="font-size:30px">{len(cocok)} area terbaik untukmu</div></div>'
         f'<div class="desk-bagian">Skor strategis (0–100) menggabungkan jarak ke {tujuan} dan sisa budget. '
         f'Hanya area dengan minimal 5 iklan yang direkomendasikan.</div></div>')

    if cocok.empty:
        termurah = dalam_radius.loc[dalam_radius["data_cukup"], "harga_standar"]
        saran = (f" Rumah 70/90 termurah dalam radius {jarak_maks} km sekitar {format_rp(termurah.min())}."
                 if not termurah.empty else " Coba perbesar jarak maksimal.")
        st.info("Belum ada area yang cocok dengan budget dan jarak ini." + saran)
    else:
        kartu = ""
        for i, b in enumerate(cocok.itertuples(), start=1):
            jarak_teks = f"{b.jarak_km:.1f}".replace(".", ",")              # 2.8 -> 2,8
            kartu += (
                '<div class="kartu-rek">'
                f'<div class="nomor">{i}</div>'
                f'<div class="rek-nama"><div class="nama-area">{b.area}</div><div class="teks-redup">{b.city} · {b.jumlah_iklan} iklan</div></div>'
                f'<div class="rek-jarak"><div class="teks-redup">Jarak</div><div class="nilai">{jarak_teks} km</div></div>'
                f'<div><div class="teks-redup">Rumah 70/90</div><div class="nilai">{format_rp(b.harga_standar)}</div>'
                f'<div class="teks-redup">sisa {format_rp(budget - b.harga_standar)}</div></div>'
                f'<div class="rek-skor"><div class="teks-redup">Skor strategis</div><div class="nilai">{b.skor:.0f}</div>'
                f'<div class="batang"><div class="isi-batang" style="width:{max(b.skor, 3):.0f}%"></div></div></div>'
                '</div>'
            )
        html(kartu)

    # 3) Peta: merah = tujuan, hitam = rekomendasi, abu = area lain dalam radius
    peta = pd.concat([
        pd.DataFrame({"lat": dalam_radius["lat"], "lon": dalam_radius["long"], "warna": "#9E9E9A", "ukuran": 120}),
        pd.DataFrame({"lat": cocok["lat"], "lon": cocok["long"], "warna": "#141414", "ukuran": 260}),
        pd.DataFrame({"lat": [t["lat"]], "lon": [t["long"]], "warna": ["#D7301F"], "ukuran": [420]}),
    ], ignore_index=True)
    st.map(peta, latitude="lat", longitude="lon", color="warna", size="ukuran", zoom=11)
    st.caption(f"🔴 {tujuan} · ⚫ area rekomendasi · ⚪ area lain dalam radius {jarak_maks} km. "
               "Jarak = garis lurus dari titik tengah area (perkiraan). Harga = estimasi model dari harga iklan.")

# ---------------------------------------------------------------------
# STRIP ANGKA + TENTANG MODEL
# ---------------------------------------------------------------------
jumlah_rumah = f"{info['jumlah_data_train'] + 631:,}".replace(",", ".")   # 2.522 latih + 631 uji
mape_teks = f"{info['mape_test_persen']:.1f}".replace(".", ",")
html(
    '<div class="strip" id="tentang">'
    f'<div class="kartu"><div class="angka">{jumlah_rumah}</div><div class="teks-redup">rumah dianalisis</div></div>'
    '<div class="kartu"><div class="angka">9</div><div class="teks-redup">kota/kabupaten Jabodetabek</div></div>'
    f'<div class="kartu"><div class="angka">±{mape_teks}%</div><div class="teks-redup">rata-rata kesalahan (MAPE)</div></div>'
    f'<div class="kartu"><div class="angka" style="font-size:22px;line-height:1.35">{info["nama_model"]}</div>'
    '<div class="teks-redup">algoritma yang dipakai versi ini</div></div></div>'
)

with st.expander("Tentang model & data"):
    st.write(f"**Model:** {info['nama_model']} · **MAPE data uji:** {info['mape_test_persen']:.2f}% · "
             f"**R² (skala log):** {info['r2_log_test']:.3f}")
    if perbandingan:                                       # tabel perbandingan metrik semua model
        tabel = pd.DataFrame(perbandingan).T
        tabel = tabel.rename(columns={"mape_test_persen": "MAPE uji (%)", "mae_test_rp": "MAE uji (Rp)",
                                      "r2_log_test": "R² uji (log)", "r2_log_train": "R² latih (log)"})
        tabel["MAE uji (Rp)"] = tabel["MAE uji (Rp)"].map(format_rp)
        st.write(f"**Perbandingan {len(tabel)} model di data uji (631 rumah):**")
        st.dataframe(tabel, use_container_width=True)
    st.write(
        "**Sumber data:** Daftar Harga Rumah Jabodetabek (Kaggle), hasil scraping iklan rumah123.com. "
        "Harga di data adalah **harga penawaran iklan**, bukan harga transaksi akhir."
    )
    st.write(
        "**Batas toleransi kewajaran** memakai MAPE model, sehingga status \"wajar\" "
        "memperhitungkan ketidakpastian prediksi."
    )

html('<div class="footer">Proyek Akhir Pembelajaran Mesin · Dini Eka Rahmawati · Siti Lativa Irmiyati · '
     'Lukiman Hakim<br>Estimasi ini bukan penilaian (appraisal) resmi.</div>')
