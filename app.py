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
    '<a href="#tentang">Tentang Model</a></div>'
    '<div class="nav-kanan"><span>ID</span><a class="pil-putih" href="#cek-harga">Mulai</a></div></div>'
)
html(
    '<div class="hero" id="beranda"><div class="hero-atas">'
    f'<div><span class="chip">{LABEL_HERO}</span><div class="judul-hero">{JUDUL_HERO}</div></div>'
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

    # 4) Peringatan jika input di luar jangkauan data latih
    if luas_bangunan > 1500 or luas_tanah > 3000:
        st.warning("Ukuran rumah ini jauh di atas sebagian besar data latih, jadi estimasi kurang akurat.")
    if luas_bangunan > luas_tanah * lantai:
        st.warning("Luas bangunan lebih besar dari luas tanah × jumlah lantai. Periksa kembali isian.")

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
    '<div class="teks-redup">model terpilih dari 3 algoritma</div></div></div>'
)

with st.expander("Tentang model & data"):
    st.write(f"**Model:** {info['nama_model']} · **MAPE data uji:** {info['mape_test_persen']:.2f}% · "
             f"**R² (skala log):** {info['r2_log_test']:.3f}")
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
