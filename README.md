# ML-A11.2023.15296-UAS

# **Deteksi Transaksi Keuangan Fraud Menggunakan Pendekatan Machine Learning dan Metode Penyeimbangan Data**

---

## **Identitas**

- **Nama**: Ravicenna Mahardhika  
- **NIM**: A11.2023.15296

---

## **2. Ringkasan dan Permasalahan project + Tujuan yang akan dicapai + Model / Alur Penyelesaian**

### Ringkasan Proyek

Penipuan transaksi digital semakin meningkat seiring dengan masifnya penggunaan layanan keuangan online. Sistem deteksi manual tidak mampu menangani volume dan variasi data yang besar dan kompleks. Oleh karena itu, pendekatan Machine Learning digunakan untuk mendeteksi transaksi fraud secara otomatis dan efisien.

Beberapa model yang digunakan dalam proyek ini meliputi:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

Masalah utama yang dihadapi adalah **ketidakseimbangan data** antara transaksi normal dan fraud. Untuk mengatasi hal ini, digunakan teknik balancing seperti:
- **SMOTE (Synthetic Minority Over-sampling Technique)** untuk menambah data pada kelas minoritas.
- **Random Undersampling** untuk mengurangi data pada kelas mayoritas.

Selain itu, digunakan **SHAP (SHapley Additive exPlanations)** sebagai alat interpretasi untuk memahami fitur-fitur yang paling berpengaruh terhadap prediksi model.

---

### Tujuan Proyek

- Membangun sistem deteksi transaksi fraud menggunakan algoritma Machine Learning.
- Menangani ketidakseimbangan data dengan teknik SMOTE dan undersampling.
- Menginterpretasi hasil model menggunakan metode SHAP untuk transparansi dan pemahaman fitur penting.


## Bagan Alur

![Bagan Alur](Dokumentasi%20Project/Img/bagan.jpg)

## **3. Penjelasan Dataset, EDA dan Proses Features Dataset**


Dataset berisi lebih dari 6 juta transaksi keuangan dengan 10 fitur utama dan 1 target isFraud. Fitur:


| Nama Kolom       | Deskripsi                                                                 |
|------------------|---------------------------------------------------------------------------|
| step             | Unit waktu simulasi (dalam jam).                                          |
| type             | Jenis transaksi (TRANSFER, CASH_OUT, DEBIT, PAYMENT, CASH_IN).           |
| amount           | Jumlah dana yang ditransaksikan.                                          |
| nameOrig         | ID akun pengirim.                                                         |
| oldbalanceOrg    | Saldo akun pengirim sebelum transaksi.                                    |
| newbalanceOrig   | Saldo akun pengirim setelah transaksi.                                    |
| nameDest         | ID akun penerima.                                                         |
| oldbalanceDest   | Saldo akun penerima sebelum transaksi.                                    |
| newbalanceDest   | Saldo akun penerima setelah transaksi.                                    |
| isFraud          | Label target (1 = fraud, 0 = normal).                                     |


EDA:

•	Distribusi target sangat imbalance:

    o	Fraud = 820 transaksi

    o	Normal = 635.440 transaksi

•	Jenis transaksi fraud hanya muncul pada TRANSFER dan CASH_OUT


Feature Engineering:

•	One-hot encoding pada type

•	Drop kolom ID

•	StandardScaler untuk normalisasi numerik


# **4. Proses Learning**

### a. Import Dataset dan Eksplorasi Awal
- Memuat dataset dan melakukan eksplorasi awal.
- Melihat distribusi kelas target (`isFraud`), statistik deskriptif fitur numerik, dan frekuensi jenis transaksi (`type`).

### b. Preprocessing Data
- Melakukan encoding pada fitur kategorik `type` agar dapat digunakan oleh model.
- Menghapus kolom ID seperti `nameOrig` dan `nameDest` karena tidak relevan untuk prediksi.
- Melakukan normalisasi pada fitur numerik agar skala data seragam.

### c. Splitting Dataset
- Membagi dataset menjadi data latih dan data uji dengan rasio 80:20.
- Menggunakan stratifikasi untuk menjaga proporsi kelas target (`isFraud`) tetap seimbang di kedua subset.

### d. Balancing Dataset
- Menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menambah jumlah data pada kelas minoritas (fraud).
- Mengombinasikan dengan undersampling pada kelas mayoritas agar dataset menjadi lebih seimbang.

### e. Training Model Machine Learning
- Melatih beberapa model klasifikasi: Logistic Regression, Random Forest, XGBoost, dan LightGBM.
- Setiap model dievaluasi untuk melihat performa dalam mendeteksi transaksi fraud.

### f. Evaluasi Model
- Menggunakan metrik evaluasi seperti:
  - Confusion Matrix
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Membandingkan performa antar model untuk memilih yang terbaik.

### g. Interpretasi Model (SHAP)
- Menggunakan SHAP (SHapley Additive exPlanations) untuk memahami fitur-fitur yang paling berpengaruh terhadap prediksi fraud.
- Visualisasi SHAP membantu menjelaskan keputusan model secara transparan.


# **5 Performa Model**

Empat model utama dilatih: **Logistic Regression**, **Random Forest**, **XGBoost**, dan **LightGBM**. Evaluasi dilakukan pada dua skenario balancing: SMOTE dan Random Undersampling.

---

### a. SMOTE Oversampling

![image.jpg](Dokumentasi%20Project/Img/sebelum.jpg)

- **Sebelum SMOTE**:  
  - Kelas 0 (Normal): 635,440  
  - Kelas 1 (Fraud): 820  

- **Setelah SMOTE**:  
  - Kelas 0 (Normal): 635,440  
  - Kelas 1 (Fraud): 635,440  

![image.jpg](Dokumentasi%20Project/Img/setelahsmote.jpg)

#### Evaluasi Model pada Data SMOTE

| Model             | Precision | Recall  | F1-score | ROC-AUC |
|------------------|-----------|---------|----------|---------|
| Logistic Reg.     | 0.0261    | 0.9823  | 0.0508   | ~0.97   |
| Random Forest     | 0.0299    | 0.9817  | 0.0580   | ~0.98   |
| XGBoost           | 0.0475    | 0.9951  | 0.0904   | ~0.99   |
| LightGBM          | 0.0272    | 0.9909  | 0.0530   | ~0.98   |

![image.jpg](Dokumentasi%20Project/Img/rocsmote.jpg)

 **XGBoost** unggul dengan presisi, recall, dan F1-score tertinggi, serta AUC terbesar (~0.9984), menunjukkan performa terbaik dalam klasifikasi. ROC curve juga memperkuat keunggulan XGBoost dibanding model lainnya.

###  Confusion Matrix – XGBoost (SMOTE)

![image.jpg](Dokumentasi%20Project/Img/confu.jpg)


|                        | Prediksi: Bukan Fraud (0) | Prediksi: Fraud (1) |
|------------------------|---------------------------|----------------------|
| **Aktual: Bukan Fraud (0)** | 1,263,746 (True Negative)   | 7,135 (False Positive) |
| **Aktual: Fraud (1)**       | 6 (False Negative)          | 1,637 (True Positive)  |

---

###  Interpretasi:

- **True Negative (TN)** = 1,263,746  
  → Model berhasil mengenali transaksi normal dengan benar.

- **False Positive (FP)** = 7,135  
  → Model salah mendeteksi transaksi normal sebagai fraud (false alarm).

- **False Negative (FN)** = 6  
  → Model gagal mendeteksi transaksi fraud (jumlahnya sangat kecil).

- **True Positive (TP)** = 1,637  
  → Model berhasil mendeteksi transaksi fraud dengan benar.

---

###  Analisis:

- **Recall sangat tinggi**:  
  Hanya 6 dari 1,643 transaksi fraud yang tidak terdeteksi → model sangat sensitif terhadap fraud.

- **Precision masih rendah**:  
  Karena ada 7.135 transaksi normal yang salah diklasifikasikan sebagai fraud.

- **Trade-off yang umum dalam deteksi fraud**:  
  Lebih baik menghasilkan false alarm daripada melewatkan transaksi penipuan yang sebenarnya.

---


### b. Random Undersampling

- **Setelah Undersampling**:  
  - Kelas 0 (Normal): 6,570  
  - Kelas 1 (Fraud): 6,570  

![image.jpg](Dokumentasi%20Project/Img/setelahunder.jpg)

####  Evaluasi Model pada Data Undersampling

| Model             | Precision | Recall  | F1-score | ROC-AUC |
|------------------|-----------|---------|----------|---------|
| Logistic Reg.     | 0.0168    | 0.9337  | 0.0331   | ~0.93   |
| Random Forest     | 0.0309    | 0.9757  | 0.0599   | ~0.96   |
| XGBoost           | 0.0553    | 0.9927  | 0.1047   | ~0.98   |
| LightGBM          | 0.0272    | 0.9750  | 0.0530   | ~0.96   |

![image.jpg](Dokumentasi%20Project/Img/rocunder.jpg)

 Pada data undersampling, **XGBoost** kembali menunjukkan performa terbaik dengan F1-score tertinggi (0.1047) dan AUC paling besar (~0.9988), menandakan keseimbangan optimal antara presisi dan recall.

---

## Interpretasi Model dengan SHAP

![image.jpg](Dokumentasi%20Project/Img/shap.jpg)

Visualisasi SHAP digunakan untuk memahami bagaimana fitur memengaruhi prediksi model terhadap satu transaksi.

- Grafik SHAP menunjukkan kontribusi setiap fitur terhadap nilai prediksi (log-odds).
- Warna **merah** menunjukkan fitur yang meningkatkan kemungkinan fraud.
- Warna **biru** menunjukkan fitur yang menurunkan kemungkinan fraud.

###  Contoh Interpretasi:
- Fitur `type_TRANSFER` dan `newbalanceOrig` (merah) mendorong model menganggap transaksi ini sebagai penipuan.
- Fitur `oldbalanceOrg`, `amount`, dan `type_CASH_OUT` (biru) justru menurunkan kemungkinan tersebut.
- Nilai akhir model sebesar **-16.37** menunjukkan keyakinan kuat bahwa transaksi ini **BUKAN** penipuan (karena sangat jauh di sisi biru).

---

##  Simulasi Prediksi

Simulasi dilakukan untuk memprediksi apakah suatu transaksi merupakan fraud berdasarkan input fitur. Hasil prediksi kemudian dianalisis menggunakan SHAP untuk melihat fitur dominan yang memengaruhi keputusan model.


## 6. Diskusi Hasil dan Kesimpulan

###  Diskusi

- Ketidakseimbangan data menyebabkan model rawan bias terhadap kelas mayoritas (non-fraud).
- Penggunaan **SMOTE** terbukti efektif dalam meningkatkan nilai **recall** dan **F1-score**, terutama pada kelas minoritas (fraud).
- **XGBoost** menunjukkan performa paling akurat dan stabil di antara semua model, menjadikannya kandidat kuat untuk implementasi sistem deteksi fraud secara real-time.
- Interpretasi menggunakan **SHAP** mengungkap bahwa fitur `type_TRANSFER` dan `newbalanceOrig` memiliki pengaruh besar dalam keputusan model.
- Simulasi dengan input ekstrem berhasil dikenali sebagai fraud, membuktikan bahwa model mampu menangkap pola-pola anomali secara efektif.

---

###  Kesimpulan

1. Algoritma Machine Learning berhasil digunakan untuk mendeteksi transaksi fraud dengan tingkat akurasi yang tinggi.
2. Teknik **SMOTE** lebih unggul dibanding **undersampling** dalam menjaga informasi penting dari data asli.
3. Kombinasi **XGBoost + SMOTE** memberikan hasil terbaik dalam hal presisi, recall, dan F1-score.
4. Penggunaan **SHAP** sangat membantu dalam menginterpretasi model yang bersifat black-box, meningkatkan transparansi dan kepercayaan terhadap sistem.
