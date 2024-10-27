#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load data
file_path = 'dataKasus-1.xlsx'  
df = pd.read_excel(file_path, sheet_name='2022')

# Tampilkan beberapa baris pertama
df.head()


# In[13]:


# Hapus kolom yang tidak diperlukan
df_cleaned = df.drop(columns=['Unnamed: 12'])

# Konversi kolom 'USIA' menjadi numerik
df_cleaned['USIA'] = df_cleaned['USIA'].str.extract('(\d+)').astype(float)

# Standarisasi kolom biner menjadi format 1/0
binary_cols = ['RIW HIPERTENSI', 'RIW PE', 'OBESITAS', 'RIW DM', 'RIW HIPERTENSI/PE DALAM KELUARGA', 'SOSEK RENDAH', 'PE/Non PE']
df_cleaned[binary_cols] = df_cleaned[binary_cols].applymap(lambda x: 1 if x == 'Ya' or x == 'PE' else 0)

# Tampilkan data setelah dibersihkan
df_cleaned.head()


# In[15]:


# Hanya memilih kolom numerik untuk analisis korelasi
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Korelasi antar variabel numerik
correlation_matrix = numeric_cols.corr()

# Menampilkan matriks korelasi
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Antar Variabel')
plt.show()


# In[16]:


##Mengisi (Imputing) missing values: Mengisi missing values dengan nilai seperti rata-rata, median, atau modus.
from sklearn.impute import SimpleImputer

# Membuat imputer untuk mengisi missing values dengan nilai rata-rata
imputer = SimpleImputer(strategy='mean')

# Mengisi missing values pada kolom yang digunakan untuk clustering
X = df_cleaned[['USIA', 'PARITAS']]
X_imputed = imputer.fit_transform(X)

# Melakukan clustering dengan K-Means pada data yang sudah diisi
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(X_imputed)

# Visualisasi hasil clustering
plt.figure(figsize=(8,6))
plt.scatter(X_imputed[:, 0], X_imputed[:, 1], c=df_cleaned['Cluster'], cmap='viridis')
plt.xlabel('USIA')
plt.ylabel('PARITAS')
plt.title('Clustering berdasarkan USIA dan PARITAS')
plt.show()


# In[17]:


# Definisikan fitur dan label (target)
X = df_cleaned[['USIA', 'PARITAS', 'RIW HIPERTENSI', 'OBESITAS']]
y = df_cleaned['PE/Non PE']

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred))

# Visualisasi pentingnya fitur dalam model
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.plot(kind='barh', title='Feature Importance')
plt.show()


# ##
