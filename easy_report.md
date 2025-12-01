# Laporan Proyek: Prediksi Gagal Bayar Pinjaman LendingClub

**Nama:** Bening Ranum
**Tanggal:** 1 Desember 2025

---

## **Pendahuluan**

Kegagalan nasabah dalam membayar pinjaman (gagal bayar atau *default*) merupakan salah satu risiko finansial terbesar bagi institusi pemberi pinjaman seperti LendingClub. Kemampuan untuk memprediksi secara akurat pinjaman mana yang berpotensi menjadi "pinjaman macet" (*bad loan*) sebelum disetujui adalah kunci untuk meminimalkan kerugian dan menjaga kesehatan portofolio kredit. Proyek ini bertujuan untuk membangun sebuah model *machine learning* yang efektif untuk mengidentifikasi pinjaman berisiko tinggi tersebut.

Secara eksplisit, proyek ini menggunakan data historis aplikasi pinjaman sebagai **input**. Fitur-fitur input ini mencakup berbagai informasi demografis dan finansial peminjam (seperti `sub_grade`, `term`, `verification_status`, jumlah pinjaman, pendapatan tahunan, dll.). **Output** dari model adalah prediksi biner: `1` untuk pinjaman yang diprediksi akan gagal bayar, dan `0` untuk pinjaman yang akan lunas.

Perjalanan proyek ini berfokus pada pengembangan bertahap dan metodis, mulai dari pemahaman bisnis dan penetapan metrik tujuan, persiapan, eksplorasi, dan *preprocessing* data, yang berakhir pada fitting model dengan cross-validation dan evaluasi model. Baseline model diambil dari *dummy* model, sementara model pembandingnya menggunakan `Random Forest`, `AdaBoost`, dan `LightGBM`. Strategi optimasi yang digunakan berpusat pada recall dan F1-score agar dapat lebih menyeimbangkan antara prediksi gagal bayar dan pinjaman lunas.

---

## **Studi Terkait**

Prediksi risiko kredit adalah area yang telah banyak diteliti dalam dunia keuangan dan ilmu data. Secara umum, pendekatan yang ada dapat dikelompokkan ke dalam beberapa kategori:

1. **Model Statistik Klasik**: Pendekatan awal sering kali menggunakan model seperti *Logistic Regression*. Kelebihan utamanya adalah interpretabilitas yang tinggi, di mana pengaruh setiap fitur terhadap probabilitas gagal bayar dapat dipahami dengan mudah. Namun, kekurangannya adalah model ini sering kali kurang akurat karena mengasumsikan hubungan linear antara fitur dan target, yang jarang terjadi pada data dunia nyata yang kompleks.

2. **Model Ensemble Berbasis Pohon (Tree-Based Ensemble)**: Ini adalah pendekatan modern yang sangat populer dan terbukti efektif untuk data tabular seperti data kredit. Algoritma seperti Random Forest, AdaBoost, dan Gradient Boosting (termasuk implementasi canggih seperti XGBoost dan LightGBM) masuk dalam kategori ini. Kelebihannya adalah kemampuan menangkap pola non-linear yang kompleks dan menghasilkan akurasi yang sangat tinggi. Kekurangannya adalah model ini cenderung bersifat (*black box*) sehingga lebih sulit diinterpretasikan. Proyek ini berada dalam kategori ini, dengan fokus pada perbandingan antara AdaBoost, Random Forest, dan LightGBM.

3. **Model Deep Learning**: Penggunaan (*neural networks*) untuk data tabular juga mulai banyak dieksplorasi. Model ini berpotensi menemukan pola yang sangat rumit yang mungkin terlewat oleh model lain. Namun, mereka memerlukan dataset yang sangat besar, *tuning* yang kompleks, dan sumber daya komputasi yang signifikan.

---

## **Dataset**

Dataset yang digunakan dalam proyek ini adalah data pinjaman publik dari **LendingClub**. Dataset ini berisi informasi lengkap tentang setiap pinjaman yang dikeluarkan, termasuk apakah pinjaman tersebut lunas atau gagal bayar, beserta puluhan fitur terkait peminjam dan pinjaman itu sendiri. Data dibagi menjadi dua set; pelatihan (*training*), dan pengujian (*test*) untuk memastikan evaluasi model yang objektif, dengan proporsi 80:20.

**Preprocessing Data** yang dilakukan merupakan bagian inti dari proyek ini:

* **Penanganan *Missing Values***: Fitur yang memiliki lebih dari 90% missing values akan dihapus. Fitur numerik yang memiliki missing values akan diimputasi dengan nilai median. Fitur kategorikal yang memiliki missing values akan diimputasi dengan nilai mode.
* **Penanganan Fitur Kategorikal**: Fitur-fitur kategorikal dibagi menjadi dua jenis: fitur dengan kardinalitas tinggi dan rendah. Fitur dengan kardinalitas rendah di-encode dengan One-Hot Encoder, sedangkan untuk fitur dengan kardinalitas rendah di-encode dengan Target Encoder.

**Contoh Sampel Dataset:**

| loan_amnt | term | int_rate | grade | home_ownership | annual_inc | loan_status (Target) |
| :-------- | :--- | :------- | :---- | :------------- | :--------- | :------------------- |
| 5000 | 36 months | 10.65% | B | RENT | 24000 | 0 (Lunas) |
| 2500 | 60 months | 15.27% | C | RENT | 30000 | 1 (Gagal Bayar) |
| 12000 | 36 months | 13.49% | C | OWN | 65000 | 0 (Lunas) |
| 20000 | 36 months | 7.90% | A | MORTGAGE | 85000 | 0 (Lunas) |

Fitur-fitur utama yang digunakan mencakup karakteristik pinjaman (`loan_amnt`, `term`, `int_rate`), informasi kredit peminjam (`grade`, `sub_grade`, `dti`), dan status pekerjaan/rumah (`emp_length`, `home_ownership`).

---

## **Metode**

Proyek ini menggunakan pendekatan *supervised machine learning* untuk masalah klasifikasi biner. Tiga algoritma utama dieksplorasi dan dibandingkan.

1. **AdaBoost (Adaptive Boosting)**

    AdaBoost bekerja secara sekuensial dengan membangun serangkaian model "lemah" (*weak learners*), yang dalam kasus ini adalah *Decision Trees*. Setiap model baru dalam urutan ini dibuat untuk memberikan perhatian lebih pada sampel-sampel yang salah diklasifikasikan oleh model sebelumnya. Bobot sampel yang salah ditebak akan ditingkatkan, sehingga "memaksa" model berikutnya untuk fokus memperbaikinya. Prediksi akhir adalah kombinasi terbobot dari semua *weak learners*.

2. **Random Forest**

    Random Forest adalah algoritma *ensemble* yang bekerja secara paralel. Algoritma ini membangun banyak *Decision Tree* secara independen satu sama lain. Setiap pohon dilatih pada sampel acak dari data (menggunakan *bootstrapping*) dan hanya mempertimbangkan subset acak dari total fitur pada setiap percabangan (*split*). Pendekatan ganda ini (acak pada data dan fitur) membantu mengurangi korelasi antar pohon dan mencegah *overfitting*. Prediksi akhir diperoleh melalui *voting* (pemungutan suara mayoritas) dari semua pohon. Model ini dikenal tangguh dan sering kali memberikan performa dasar yang sangat kuat dengan sedikit *tuning*.

3. **LightGBM (Light Gradient Boosting Machine)**

    LightGBM adalah implementasi dari *Gradient Boosting* yang sangat efisien dan berkinerja tinggi. Seperti *boosting* lainnya, ia membangun pohon secara sekuensial di mana setiap pohon baru mencoba memperbaiki kesalahan (residual) dari pohon sebelumnya. Keunikan LightGBM terletak pada dua teknik utama:

    * **Gradient-based One-Side Sampling (GOSS)**: Alih-alih menggunakan semua sampel data untuk menghitung gradien (kesalahan), GOSS mempertahankan semua sampel dengan gradien besar (yang paling banyak salah) dan mengambil sampel acak dari yang bergradien kecil. Ini mempercepat pelatihan tanpa banyak mengorbankan akurasi.
    * **Exclusive Feature Bundling (EFB)**: Menggabungkan fitur-fitur yang jarang aktif secara bersamaan (*mutually exclusive*) menjadi satu fitur tunggal untuk mengurangi dimensi data.

    Selain itu, LightGBM menumbuhkan pohon secara *leaf-wise* (memilih daun yang akan memberikan pengurangan *loss* terbesar), yang sering kali menghasilkan model yang lebih akurat dibandingkan pertumbuhan *level-wise* tradisional. Parameter penting yang saya gunakan adalah `scale_pos_weight`, yang digunakan untuk memberikan bobot lebih pada kelas minoritas (pinjaman macet) untuk mengatasi ketidakseimbangan kelas.

---

## **Uji Coba**

Eksperimen dilakukan secara iteratif. *Hyperparameter* yang dipilih dieksplorasi menggunakan pencarian grid (*random search*) dengan validasi silang (*cross-validation*) 3-fold untuk mencegah *overfitting* ,mendapatkan estimasi performa yang stabil, dan mempersingkat waktu latih.

Fokus utama terletak pada *tuning hyperparameter* yang krusial, seperti `learning_rate` dan `n_estimators` untuk model *boosting*, yang mengontrol trade-off antara kecepatan belajar dan kompleksitas model. Untuk LightGBM, saya secara spesifik bereksperimen dengan parameter `scale_pos_weight` untuk secara langsung menargetkan peningkatan *recall*.

Metrik evaluasi utama yang digunakan adalah **ROC-AUC**, **Precision**, dan **Recall** untuk kelas positif (pinjaman macet).

* **ROC-AUC**: Mengukur kemampuan model secara keseluruhan untuk membedakan antara kelas positif dan negatif, tidak terpengaruh oleh ambang batas (*threshold*) klasifikasi.
* **Recall**: Dari semua pinjaman yang sebenarnya macet, berapa persen yang berhasil diidentifikasi oleh model? Metrik ini menjadi fokus utama karena tujuan bisnis adalah meminimalkan risiko dengan menangkap sebanyak mungkin pinjaman buruk.
* **Precision**: Dari semua pinjaman yang diprediksi macet, berapa persen yang benar-benar macet? Metrik ini penting untuk memastikan model tidak terlalu banyak menolak nasabah baik yang sebenarnya layak mendapatkan pinjaman.

## **Hasil Eksperimen**

Hasil performa dari model-model terbaik pada **Test Set** dirangkum dalam tabel berikut:

| Model / Konfigurasi | ROC-AUC | Precision (Pinjaman Macet) | Recall (Pinjaman Macet) |
| :---------------------- | :------ | :------------------------- | :---------------------- |
| AdaBoost (Recall-Optimized) | 0.665 | 0.30 | 0.75 |
| RandomForest (Balanced) | 0.702 | 0.40 | 0.44 |
| LGBM-Conservative (Highest AUC) | **0.717** | **0.56** | 0.13 |
| LGBM-Aggressive (Recall-Optimized) | 0.711 | 0.25 | **0.95** |

<center>

![Perbandingan Kurva ROC](/assets/img_01.png)

</center>

<center>

Grafik ini mengukur kemampuan model dalam pemisahan pinjaman baik dari pinjaman macet.

![Perbandingan Kurva Precision-Recall](/assets/img_02.png)

</center>

Grafik ini mengukur efisiensi bisnis, melihat trade-off antara Recall dan Precision.

* **AdaBoost** terbukti merupakan model dengan daya prediksi yang lemah (diperlihatkan oleh skor AUC terendah (0.665)). Menariknya, kurva Precision-Recall menunjukkan bahwa AdaBoost memiliki skor Average Precision (AP) tertinggi. Namun, bentuk kurvanya yang aneh—presisi sempurna pada recall yang sangat rendah lalu anjlok—menunjukkan bahwa model ini tidak stabil dan mungkin hanya unggul dalam mengidentifikasi segmen data yang sangat spesifik. Perilaku ini berisiko untuk aplikasi bisnis umum.

* **RandomForest** berhasil menjadi **baseline yang kuat dan andal**. Dengan AUC 0.702, model ini jauh lebih "pintar" daripada AdaBoost dan memberikan keseimbangan antara *precision* (0.40) dan *recall* (0.44) tanpa penyesuaian khusus.

* **LightGBM** muncul sebagai **mesin (*engine*) terbaik**. Kedua konfigurasi LightGBM menunjukkan kurva ROC yang hampir identik dan tertinggi, membuktikan kekuatan prediktif superior dari algoritma ini. Kurva PR untuk kedua model LGBM menunjukkan perilaku yang sehat dan konsisten. LGBM (Highest AUC) menonjol karena menawarkan presisi tertinggi pada tingkat recall yang lebih rendah, menjadikannya ideal untuk strategi bisnis yang konservatif. Eksperimen antara konfigurasi "Highest AUC" dan "Highest Recall" membuktikan fleksibilitas LightGBM. Kedua grafik mengonfirmasi bahwa keduanya berasal dari 'mesin' prediksi yang sama kuatnya (terlihat dari kurva yang hampir tumpang tindih). Perbedaan utama mereka bukanlah pada kemampuan, melainkan pada titik operasi pada kurva yang kita pilih sesuai kebutuhan bisnis.

---

## **Simpulan**

Proyek ini berhasil melakukan transformasi pada *pipeline* prediksi gagal bayar, mulai dari model AdaBoost yang berkinerja buruk menjadi model LightGBM yang canggih dan kuat. Kesimpulan utamanya adalah:

1. Meskipun AdaBoost menunjukkan Average Precision tertinggi, LightGBM (AUC=0.717, AP=0.412) terbukti sebagai model yang paling kuat dan seimbang secara keseluruhan. Kurva ROC-nya secara definitif superior, dan kurva Precision-Recall-nya menunjukkan perilaku yang stabil dan dapat diandalkan, tidak seperti AdaBoost.
2. Ada trade-off yang jelas dan dapat dikontrol antara mengidentifikasi sebanyak mungkin pinjaman macet (recall) dan memastikan prediksi tersebut akurat (precision), yang dapat disesuaikan pada model LightGBM untuk memenuhi tujuan bisnis yang spesifik.

## **Penelitian Lanjutan**

Langkah selanjutnya tidak lagi berfokus pada pencarian algoritma baru, melainkan pada optimalisasi model terbaik yang telah kita identifikasi. Jika saya memiliki sumber daya berlebih, eksplorasi selanjutnya akan mencakup:

1. **Threshold Tuning**: Menggunakan model LightGBM yang sudah dilatih, kita akan menganalisis kurva Precision-Recall dan prediksi probabilitasnya. Alih-alih menggunakan ambang batas default 0.5, kita akan memilih ambang batas baru (misalnya, 0.35) yang dapat memenuhi target *recall* bisnis (misal, 70%) sambil memaksimalkan *precision* pada tingkat *recall* tersebut.
2. **Re-tuning dengan F-beta Score**: Daripada mengoptimalkan AUC atau *recall* secara terpisah, kita dapat menjalankan kembali proses *hyperparameter tuning* dengan metrik **F-beta score** (misalnya, F2-score). Metrik ini secara matematis memberikan bobot lebih pada *recall* daripada *precision*, sehingga proses *tuning* akan secara alami mencari parameter yang menghasilkan keseimbangan yang lebih baik dan sesuai dengan tujuan bisnis.
3. **Feature Engineering Lanjutan**: Untuk meningkatkan daya prediksi inti model (menaikkan AUC lebih tinggi lagi), kita dapat mengeksplorasi pembuatan fitur-fitur baru.
