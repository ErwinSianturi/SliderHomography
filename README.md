# 📐 Homography Slider & Visual Inspector

Interactive calibration and visual inspection tool for multi-camera perspective homography transformations in retail shopper tracking and floorplan spatial analytics.

---

## 🚀 Fitur Utama

- **Dual-Panel Visualizer**: Tampilan berdampingan antara frame kamera asli (*camera view*) dengan proyeksi deteksi pada denah toko (*floorplan view*).
- **Interactive Fine-Tuning Sliders**:
  - Translasi $(\Delta X, \Delta Y)$ dalam piksel dan meter.
  - Skala Non-uniform $(S_x, S_y)$.
  - Rotasi sudut $(\theta)$ dengan pivot fleksibel.
  - *Foot Point Ratio* $(\beta)$ untuk penyesuaian posisi tapak kaki pejalan kaki.
- **Deteksi Anomali & Rekomendasi Spasial**: Registry anomali terintegrasi untuk mendeteksi kamera kritis (misal deteksi orang melompat ke lorong sebelah atau rak pembatas).
- **Dual Export Formats**:
  - **Format A (`homography_calibration_overrides.json`)**: Log audit lengkap, parameter eksperimen, matriks homografi $3 \times 3$, dan metrik akurasi spasial.
  - **Format B (`cameras_export_calibrated.json`)**: Kloning 100% kompatibel dengan pipeline produksi master.

---

## 🛠️ Instalasi & Menjalankan

### 1. Prasyarat
Pastikan Python 3.10+ sudah terpasang.

### 2. Pasang Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```
Akses di browser Anda melalui `http://localhost:8501`.

---

## 🌐 Menjalankan dengan Cloudflare Tunnel (TryCloudflare)

Untuk membagikan aplikasi secara publik agar bisa diuji oleh orang lain tanpa konfigurasi port forwarding:

1. **Jalankan Streamlit:**
   ```powershell
   streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
   ```

2. **Jalankan Tunnel:**
   ```powershell
   cloudflared tunnel --url http://localhost:8501
   ```
   Buka URL `https://<random-subdomain>.trycloudflare.com` yang dihasilkan di terminal.

---

## 🧪 Pengujian Unit & Verifikasi

Untuk menjalankan automated test suite:
```bash
pytest tests
```

---

## 📂 Struktur Repositori

```text
├── app.py                                   # Aplikasi utama Streamlit
├── core/                                    # Engine kalibrasi & matematika spasial
│   ├── config_loader.py                     # Loader metadata kamera & registry anomali
│   ├── exporter.py                          # Export handler Format A & Format B
│   ├── floorplan_renderer.py                # Visualizer denah & bounding box
│   ├── homography_engine.py                 # Algoritma DLT & transformasi matriks
│   └── spatial_validator.py                 # Validasi aisle / blocker region
├── tests/                                   # Pytest automated test suite
├── cameras_export.json                      # Konfigurasi master kamera & homography awal
├── floorplan.png                            # Peta denah toko resolusi penuh
└── homography_visual_downloads_20261023/    # Dataset frame gambar & deteksi
```
