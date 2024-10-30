#!/usr/bin/env python
# coding: utf-8

# ### MUHAMMAD FADHLAN HAKIM || A11.2022.14619

# ##### Import Libraries: 
# ###### mengimport pustaka/libraries, termasuk pandas dan numpy untuk manipulasi data, seaborn dan matplotlib untuk visualisasi, serta pustaka dari sklearn untuk pemodelan dan evaluasi.

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ##### Load Dataset: 
# ###### Mengambil dataset dari file Excel dataKasus-1.xlsx

# In[2]:


# Load Dataset from Excel
data = pd.read_excel('dataKasus-1.xlsx')  # Memuat dataset dari file Excel
print("Data Loaded Successfully!")
print("Info Data:", data.info())
print("Beberapa data awal:\n", data.head())


# ##### Data Preprocessing: 
# ###### Membersihkan data dengan cara membuang kolom yang tidak diperlukan (NO, NAMA, Unnamed: 12) serta mengisi nilai kosong (NaN) di kolom USIA dan JARAK KELAHIRAN dengan nilai yang paling sering muncul atau modus.

# In[4]:


# Data Preprocessing
print("\n---Data Preprocessing---")
# Drop irrelevant columns
data_cleaned = data.drop(columns=['NO', 'NAMA', 'Unnamed: 12'])


# In[6]:


# Handle missing values by filling with the most frequent value (mode)
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])
data_cleaned['JARAK KELAHIRAN'] = data_cleaned['JARAK KELAHIRAN'].fillna(data_cleaned['JARAK KELAHIRAN'].mode()[0])


# ##### Handle Missing Values and Data Cleaning: 
# ###### Mengisi nilai NaN di kolom USIA, menghapus karakter non-numerik seperti " TH" dari kolom tersebut, lalu mengonversinya ke tipe integer.

# In[10]:


# Isi nilai NaN dengan modus terlebih dahulu untuk menghindari masalah konversi
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])

# Remove non-numeric characters (like " TH") in the 'USIA' column
data_cleaned['USIA'] = data_cleaned['USIA'].str.extract(r'(\d+)')

# Pastikan tidak ada nilai NaN setelah proses ekstraksi angka
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])

# Konversi ke tipe integer setelah memastikan tidak ada NaN
data_cleaned['USIA'] = data_cleaned['USIA'].astype(int)


# ##### Encode Categorical Data: 
# ###### Mengonversi data kategoris menjadi format numerik menggunakan LabelEncoder untuk setiap kolom bertipe objek.

# In[11]:


# Encode categorical data
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le


# ##### Splitting Data into Features and Target: 
# ###### Memisahkan data menjadi variabel fitur (X) dan target (y) untuk persiapan pemodelan.

# In[12]:


# Splitting data into Features (X) and Target (y)
target_column_name = 'PE/Non PE'
X = data_cleaned.drop(target_column_name, axis=1)
y = data_cleaned[target_column_name]


# ##### Exploratory Data Analysis (EDA):
# ###### Melakukan analisis eksplorasi data, termasuk statistik deskriptif dan visualisasi korelasi antar fitur menggunakan heatmap.

# In[13]:


# Exploratory Data Analysis (EDA)
print("\n---Exploratory Data Analysis---")
print("Descriptive Statistics:\n", data_cleaned.describe())
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# ##### Visualisasi Distribusi Variabel Target:
# ###### Membuat visualisasi distribusi kelas target untuk memahami proporsi kelas yang akan diprediksi.

# In[14]:


# Visualisasi distribusi variabel target
plt.figure(figsize=(8, 6))
sns.countplot(data_cleaned[target_column_name])
plt.title("Distribusi Kelas Target")
plt.show()


# ##### Feature Selection Using Recursive Feature Elimination (RFE):
# ###### Memilih fitur yang paling relevan menggunakan teknik Recursive Feature Elimination (RFE) dengan model DecisionTreeClassifier.

# In[15]:


# Feature Selection (15 AH) using Recursive Feature Elimination (RFE)
print("\n---Feature Selection---")
model = DecisionTreeClassifier(random_state=42)
selector = RFE(model, n_features_to_select=15, step=1)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)


# ##### Train-test split: 
# ###### Memisahkan dataset menjadi dua bagian: data pelatihan dan data pengujian.
# ###### Tujuan: Melatih model menggunakan data pelatihan (80%) dan menguji performa model di data pengujian (20%).

# In[16]:


# Train-test split
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_selected, X_test_selected, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# ##### Standardize data: Menyamakan skala fitur agar memiliki mean 0 dan standar deviasi 1, membantu meningkatkan performa model.
# ###### scaler = StandardScaler(): Membuat objek StandardScaler dari sklearn untuk melakukan standardisasi.
# ###### X_train_full = scaler.fit_transform(X_train_full): Melatih (fit) scaler pada data pelatihan penuh (X_train_full) dan langsung mengubah (transform) datanya.
# ###### X_test_full = scaler.transform(X_test_full): Menggunakan scaler yang sudah dilatih untuk menstandarisasi data pengujian penuh (X_test_full).
# ###### X_train_selected = scaler.fit_transform(X_train_selected): Melatih scaler baru pada data pelatihan terpilih (X_train_selected) dan langsung mengubahnya.
# ###### X_test_selected = scaler.transform(X_test_selected): Menggunakan scaler yang dilatih pada data pelatihan terpilih untuk menstandarisasi data pengujian terpilih (X_test_selected).

# In[17]:


# Standardize data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)


# ##### Model Training:
# 
# ###### Melatih beberapa model, seperti Naive Bayes, K-Nearest Neighbors, dan Decision Tree, pada data latih (X_train dan y_train).
# ###### Masing-masing model dievaluasi dengan teknik validasi silang (cross-validation) untuk mendapatkan skor yang lebih stabil pada data latih.

# In[19]:


# Modeling and Evaluation with Cross-Validation
print("\n---Modeling and Evaluation---")
models = {
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Initialize dictionaries to store scores
scores_full = {}
scores_selected = {}

for name, model in models.items():
    # Original Data
    print(f"\nModel: {name} on Original Data")
    model.fit(X_train_full, y_train)
    y_pred_full = model.predict(X_test_full)
    cm_full = confusion_matrix(y_test, y_pred_full)
    print("Confusion Matrix (Original):\n", cm_full)
    print("Classification Report (Original):\n", classification_report(y_test, y_pred_full))
    cv_scores_full = cross_val_score(model, X_train_full, y_train, cv=5)
    scores_full[name] = (cv_scores_full.mean(), cv_scores_full.std())
    
    # Selected Data
    print(f"\nModel: {name} on Selected Features")
    model.fit(X_train_selected, y_train)
    y_pred_selected = model.predict(X_test_selected)
    cm_selected = confusion_matrix(y_test, y_pred_selected)
    print("Confusion Matrix (Selected):\n", cm_selected)
    print("Classification Report (Selected):\n", classification_report(y_test, y_pred_selected))
    cv_scores_selected = cross_val_score(model, X_train_selected, y_train, cv=5)
    scores_selected[name] = (cv_scores_selected.mean(), cv_scores_selected.std())


# #### Langkah ini bertujuan untuk membandingkan performa model sebelum dan sesudah seleksi fitur, dengan menampilkan skor validasi silang (cross-validation mean score) untuk setiap model pada kedua dataset.
# #### Untuk setiap model dalam models, blok kode ini menampilkan:
# ##### Original Data - Mean CV Score: 
# ###### Skor rata-rata validasi silang pada dataset lengkap sebelum seleksi fitur. Ini membantu mengevaluasi kinerja model saat semua fitur digunakan.
# ##### Selected Data - Mean CV Score: 
# ###### Skor rata-rata validasi silang pada dataset dengan fitur yang telah dipilih (melalui metode seleksi fitur seperti RFE). Skor ini menunjukkan performa model saat hanya fitur yang paling relevan yang digunakan.

# In[20]:


# Comparison Analysis
print("\n---Comparison Analysis---")
for name in models.keys():
    print(f"{name}:")
    print(f"Original Data - Mean CV Score: {scores_full[name][0]:.4f} ± {scores_full[name][1]:.4f}")
    print(f"Selected Data - Mean CV Score: {scores_selected[name][0]:.4f} ± {scores_selected[name][1]:.4f}")
    print()


# ##### Visualisasi Hasil:
# ##### Tujuan: 
# ###### Menampilkan perbandingan skor validasi silang rata-rata (Mean CV Score) dari setiap model, baik pada dataset asli (sebelum seleksi fitur) maupun dataset yang telah mengalami seleksi fitur.
# ##### Penjelasan Kode:
# ###### labels: Label untuk setiap model dalam bentuk daftar, yang akan digunakan pada sumbu x.
# ###### original_scores dan selected_scores: Dua daftar yang berisi skor rata-rata validasi silang untuk masing-masing model pada data asli (original_scores) dan data yang sudah melalui seleksi fitur (selected_scores).
# ###### x: Posisi untuk setiap model pada sumbu x, dan width menentukan lebar batang pada grafik.
# ###### Plot Batang: rects1 dan rects2 adalah batang yang merepresentasikan skor untuk data asli dan data setelah seleksi fitur. Mereka diposisikan berdampingan agar mudah dibandingkan.
# ###### Label dan Tampilan:
# ###### ax.set_ylabel memberi label pada sumbu y sebagai "Mean CV Score".
# ###### ax.set_title menambahkan judul grafik untuk memberi konteks visualisasi ini sebagai perbandingan performa model pada dataset asli vs. dataset dengan fitur terpilih.
# ###### ax.legend menampilkan legenda untuk mengidentifikasi batang yang mewakili data asli dan data yang telah dipilih.

# In[21]:


# Visualisasi Hasil
labels = list(models.keys())
original_scores = [scores_full[model][0] for model in labels]
selected_scores = [scores_selected[model][0] for model in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_scores, width, label='Original Data')
rects2 = ax.bar(x + width/2, selected_scores, width, label='Selected Features')

ax.set_ylabel('Mean CV Score')
ax.set_title('Comparison of Model Performance on Original vs Selected Features')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# In[ ]:




