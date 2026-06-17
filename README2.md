# Klasifikasi Kanker Kulit (Benign vs Malignant)
## Nama Anggota
###  Anggota A : Septania Sybil Shofiyah (F1D02410094)
###  Anggota B : I Gde Surya Laksana Diputra (F1D02410051)
###  Anggota C : Khofipah Indar Putri (F1D02410061)
###  Anggota D : Islam Ahmed Fouad Abunima (F1D0241100-)

# Project Overview
Pada project ini, dilakukan eksperimen klasifikasi citra kanker kulit ke dalam dua kelas, yaitu **Benign** dan **Malignant**, menggunakan dataset yang berisi 160 gambar untuk data training (80 gambar per label) dan 216 gambar untuk data testing tanpa label. Tujuan dari project ini adalah menguji kemampuan dalam mengimplementasikan teknik pengolahan citra digital untuk klasifikasi, serta memilih tahapan preprocessing yang tepat sesuai karakteristik data kanker kulit.

Dataset kanker kulit dipilih karena tekstur, kontras, dan bentuk lesi pada permukaan kulit menjadi indikator penting dalam membedakan lesi yang bersifat jinak (benign) dengan yang berpotensi ganas (malignant). Karakteristik visual seperti tepi lesi yang tidak beraturan, variasi warna, dan tekstur permukaan menjadi dasar pemilihan teknik preprocessing dan ekstraksi fitur yang digunakan sepanjang project ini.

Eksperimen dilakukan sebanyak 4 kali percobaan (Percobaan 0 sampai 3) dengan kombinasi preprocessing yang berbeda-beda, untuk melihat bagaimana setiap teknik mempengaruhi hasil akurasi klasifikasi dari tiga model, yaitu Random Forest, SVM, dan KNN.

# IMPORT LIBRARY
Seluruh percobaan menggunakan kombinasi library berikut:

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
```

`cv2` dipakai untuk operasi pengolahan citra (resize, grayscale, thresholding, dll), `numpy` untuk manipulasi array dan piksel, `pandas` untuk menyusun hasil ekstraksi fitur ke dalam tabel, `skimage` untuk menghitung matriks GLCM, dan `sklearn` untuk keperluan splitting data, normalisasi, serta pemodelan klasifikasi.

# Load Data
Dataset diorganisasikan dalam struktur folder sebagai berikut:

```
.
└── dataset
    └── Train
        ├── benign
        │   ├── image1.jpg
        │   └── dst...
        └── malignant
            └── dst...
    └── Test
        └── (216 gambar tanpa label)
```

Proses loading dilakukan dengan membaca seluruh gambar pada masing-masing folder label, lalu menyimpan citra, label, dan nama file ke dalam list `data`, `labels`, dan `file_name`. Ukuran gambar diseragamkan menjadi 256x256 piksel menggunakan resize agar konsisten sebelum masuk ke tahap preprocessing dan ekstraksi fitur.

## Data Understanding
Dataset training terdiri dari 160 gambar dengan distribusi seimbang, yaitu 80 gambar untuk label Benign dan 80 gambar untuk label Malignant. Secara visual, gambar kanker kulit memiliki variasi pencahayaan, warna kulit, serta bentuk dan ukuran lesi yang berbeda-beda antar gambar. Karakteristik ini menjadi pertimbangan dalam memilih preprocessing, terutama untuk menonjolkan tekstur dan kontras lesi agar dapat dibedakan dengan baik melalui fitur GLCM.

Data testing berjumlah 216 gambar, namun tidak memiliki subfolder label sehingga tidak bisa digunakan untuk menghitung akurasi secara langsung. Data ini lebih difungsikan sebagai data prediksi murni pada beberapa percobaan.

# Data Preparation
## Data Augmentation
Augmentasi data tidak dilakukan pada seluruh percobaan, karena jumlah data per label (80 gambar) sudah memenuhi syarat minimal rentang 70-100 gambar yang ditentukan. Variabel `data_augmented` pada beberapa percobaan hanya merupakan salinan langsung dari `data` tanpa proses augmentasi tambahan.

## Preprocessing
Keempat percobaan menggunakan kombinasi preprocessing yang berbeda, dengan tingkat kompleksitas yang meningkat secara bertahap dari Percobaan 0 hingga Percobaan 3.

### Percobaan 0: Resize + Grayscale
Percobaan paling dasar ini hanya menerapkan dua tahap preprocessing, yaitu resize ukuran gambar menjadi 256x256 piksel, kemudian konversi ke grayscale.

```python
def resize_grayscale(img):
    img_resize = cv2.resize(img, (256,256))
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    return img_gray
```

Tujuan dari percobaan ini adalah membangun baseline performa model tanpa preprocessing tambahan, sehingga bisa dijadikan pembanding terhadap percobaan-percobaan berikutnya yang menggunakan lebih banyak tahap preprocessing.

### Percobaan 1: Resize + Median Filter + Histogram Equalization
Percobaan kedua menambahkan dua tahap preprocessing setelah resize, yaitu median filter untuk mengurangi noise, dan histogram equalization untuk meningkatkan kontras citra.

```python
def prepro1(img_gray):
    """Median Filter -> menghilangkan noise pada gambar."""
    return cv2.medianBlur(img_gray, 5)

def prepro2(img_gray):
    """Histogram Equalization -> meningkatkan kontras gambar."""
    return cv2.equalizeHist(img_gray)
```

Pipeline yang dijalankan adalah Resize → Median Filter → Histogram Equalization. Median filter dipilih karena efektif menghilangkan noise berupa bintik-bintik kecil pada citra dermoskopi tanpa merusak tepi lesi, sementara histogram equalization membantu menonjolkan detail tekstur lesi yang mungkin tersembunyi akibat kontras yang rendah.

### Percobaan 2: Grayscale + Sharpening + Thresholding
Percobaan ketiga menggunakan tiga tahap preprocessing yang seluruhnya custom, di mana sharpening dan thresholding diimplementasikan secara manual piksel per piksel.

```python
def sharpening_manual(img_gray):
    """Sharpening menggunakan kernel Laplacian 3x3, dihitung manual piksel per piksel."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    # ... konvolusi manual ...

def thresholding_manual(img_gray, batas=127):
    """Binerisasi manual: piksel > batas jadi 255, selain itu jadi 0."""
    # ... loop piksel per piksel ...
```

Pipeline pada percobaan ini adalah Grayscale → Sharpening → Thresholding, dijalankan secara bertahap (hasil grayscale menjadi input sharpening, hasil sharpening menjadi input thresholding). Sharpening manual dengan kernel Laplacian digunakan untuk mempertegas tepi lesi, sementara thresholding manual mengubah citra menjadi biner untuk menyederhanakan bentuk lesi yang akan diekstraksi fiturnya.

### Percobaan 3: Otsu Thresholding + Opening + Closing
Percobaan keempat menggunakan kombinasi morfologi citra, yaitu thresholding otomatis dengan metode Otsu, dilanjutkan dengan operasi morfologi opening dan closing menggunakan kernel 5x5.

```python
def percobaan3(img):
    _, img_tresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_opening = opening(img_tresh, kernel)
    img_closing = closing(img_opening, kernel)
    return img_closing
```

Otsu thresholding dipilih karena dapat menentukan nilai ambang batas secara otomatis berdasarkan distribusi histogram citra, sehingga lebih adaptif dibandingkan thresholding manual dengan nilai batas tetap. Operasi opening digunakan untuk menghilangkan noise kecil pada hasil threshold, sedangkan closing digunakan untuk menutup celah kecil pada area lesi, sehingga bentuk objek menjadi lebih utuh dan rapi sebelum diekstraksi fiturnya.

## Feature Extraction
Keempat percobaan secara konsisten menggunakan metode Gray Level Co-occurrence Matrix (GLCM) untuk ekstraksi fitur, dengan sudut 0, 45, 90, dan 135 derajat, bersifat simetris, dan distance 1.

```python
def glcm(image, derajat):
    if derajat == 0:
        angles = [0]
    elif derajat == 45:
        angles = [np.pi / 4]
    elif derajat == 90:
        angles = [np.pi / 2]
    elif derajat == 135:
        angles = [3 * np.pi / 4]

    glcm = graycomatrix(image, [1], angles, 256, symmetric=True, normed=True)
    return glcm
```

Fitur yang diekstraksi dari setiap sudut meliputi Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Entropy, dan ASM, sehingga total terdapat 28 fitur (7 fitur x 4 sudut) untuk Percobaan 1 dan 3, sedangkan pada Percobaan 0 dan 2 fitur dirata-rata dari 4 sudut menjadi 4 fitur utama (Contrast, Correlation, Energy, Homogeneity).

## Feature Selection
Pada Percobaan 1 dan 3, seleksi fitur dilakukan menggunakan metode correlation, yaitu menyaring fitur yang memiliki korelasi absolut tinggi (>0.95 untuk Percobaan 1, >1.00 untuk Percobaan 3) terhadap fitur lain, untuk menghindari redundansi informasi antar fitur.

```python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
threshold = 0.95
# ... penyaringan fitur berdasarkan korelasi ...
```

Sementara itu, Percobaan 0 dan 2 tidak melakukan seleksi fitur tambahan karena jumlah fitur yang dihasilkan (4 fitur) sudah cukup ringkas.

## Splitting Data
Seluruh percobaan menggunakan rasio pembagian data 80:20 (80% data training, 20% data testing/validasi) dengan `random_state=42` agar hasil pembagian data konsisten dan dapat direproduksi.

```python
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)
```

Pada Percobaan 2, karena folder Test asli tidak memiliki label, splitting dilakukan terhadap data Train saja untuk keperluan evaluasi, sementara data Test asli digunakan murni untuk prediksi tanpa perhitungan akurasi.

## Normalization
Seluruh percobaan menggunakan teknik normalisasi standarization (Z-score), yaitu mengurangi nilai fitur dengan rata-rata dan membaginya dengan standar deviasi data training.

```python
X_test = (X_test - X_train.mean()) / X_train.std()
X_train = (X_train - X_train.mean()) / X_train.std()
```

Normalisasi ini penting terutama untuk model SVM dan KNN yang sensitif terhadap skala antar fitur, sehingga fitur dengan rentang nilai besar tidak mendominasi proses pembelajaran model.

# Modeling
Ketiga model klasifikasi yang digunakan secara konsisten di seluruh percobaan adalah:

```python
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
```

Catatan: pada Percobaan 1 dan 3, jumlah estimator Random Forest yang digunakan adalah 5, sedangkan pada Percobaan 0 dan 2 digunakan 100 estimator.

# Evaluation
Berikut adalah hasil akurasi dari setiap model pada masing-masing percobaan, dihitung dari data validasi/testing yang memiliki label asli.

| Percobaan | Preprocessing | Random Forest | SVM | KNN |
|---|---|---|---|---|
| Percobaan 0 | Resize + Grayscale | - | - | 59.38% |
| Percobaan 1 | Resize + Median Filter + Histogram Equalization | 75% | 69% | 78% |
| Percobaan 2 | Grayscale + Sharpening + Thresholding (manual) | 59.38% | 62.50% | 59.38% |
| Percobaan 3 | Otsu Thresholding + Opening + Closing | 84.38% | 78.13% | 84.38% |

Confusion matrix ditampilkan pada setiap percobaan untuk melihat distribusi kesalahan klasifikasi antara kelas Benign dan Malignant, yang penting dalam konteks medis karena kesalahan memprediksi Malignant sebagai Benign berisiko lebih besar dibandingkan kesalahan sebaliknya.

# Kesimpulan

Dari keempat percobaan yang dilakukan, terlihat bahwa pemilihan kombinasi preprocessing memberikan pengaruh yang cukup signifikan terhadap performa model klasifikasi, meskipun fitur yang diekstraksi sama-sama menggunakan metode GLCM.

Percobaan 0 yang hanya menggunakan resize dan grayscale sebagai baseline menghasilkan akurasi yang rendah (sekitar 59%), menunjukkan bahwa preprocessing minimal belum cukup untuk menonjolkan karakteristik tekstur yang membedakan lesi benign dan malignant.

Percobaan 1 dengan tambahan median filter dan histogram equalization menunjukkan peningkatan akurasi yang cukup baik pada model KNN (78%), mengindikasikan bahwa peningkatan kontras melalui histogram equalization cukup membantu memperjelas perbedaan tekstur antar kelas.

Percobaan 2 yang menggunakan sharpening dan thresholding manual justru menghasilkan akurasi yang relatif rendah dan mirip dengan baseline (sekitar 59-62%). Hal ini kemungkinan disebabkan oleh proses thresholding yang mengubah citra menjadi biner penuh (hitam putih), sehingga banyak informasi gradasi intensitas yang hilang dan menyebabkan fitur GLCM antar kelas menjadi kurang diskriminatif.

Percobaan 3 yang menggunakan Otsu thresholding dikombinasikan dengan operasi morfologi opening dan closing memberikan hasil terbaik di antara seluruh percobaan, dengan akurasi mencapai 84.38% pada Random Forest dan KNN. Hal ini menunjukkan bahwa thresholding otomatis berbasis Otsu lebih adaptif dalam menentukan batas segmentasi dibandingkan thresholding manual dengan nilai tetap, dan operasi morfologi membantu membersihkan noise serta merapikan bentuk lesi sebelum ekstraksi fitur, sehingga fitur tekstur yang dihasilkan menjadi lebih representatif.

Secara keseluruhan, dapat disimpulkan bahwa tidak semua preprocessing yang lebih kompleks otomatis menghasilkan akurasi yang lebih baik. Pemilihan jenis preprocessing harus disesuaikan dengan karakteristik citra yang diolah, di mana untuk dataset kanker kulit, preprocessing yang mempertahankan informasi tekstur sekaligus melakukan segmentasi yang adaptif (seperti Otsu thresholding dan operasi morfologi pada Percobaan 3) terbukti lebih efektif dibandingkan preprocessing yang terlalu menyederhanakan citra menjadi biner secara kasar (seperti thresholding manual pada Percobaan 2).