#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


# 1. Memuat dataset
df = pd.read_csv('mushroom_cleaned.csv')


# In[3]:


# 2. Preprocessing: Mengatasi missing values (jika ada)
# Memeriksa apakah ada nilai kosong
print(df.isnull().sum())


# In[4]:


# Mengisi nilai kosong dengan metode pengisian yang sesuai (median, modus, dll.)
# Sebagai contoh, jika ada nilai kosong, kita dapat menggunakan:
df.fillna(df.median(), inplace=True)


# In[5]:


# 3. Label Encoding untuk kolom kategorikal yang tidak banyak variasi
label_encoder = LabelEncoder()


# In[6]:


# Melakukan label encoding pada kolom 'cap-shape' dan 'gill-color'
df['cap-shape'] = label_encoder.fit_transform(df['cap-shape'])
df['gill-color'] = label_encoder.fit_transform(df['gill-color'])


# In[7]:


# 4. One-Hot Encoding untuk kolom kategorikal lainnya
# Mengidentifikasi kolom kategorikal untuk di-encode
categorical_columns = ['gill-attachment', 'stem-color', 'season']
one_hot_encoder = OneHotEncoder(sparse_output=False)


# In[8]:


# Menerapkan OneHotEncoder
encoded_categorical = one_hot_encoder.fit_transform(df[categorical_columns])


# In[9]:


# Menambah hasil OneHotEncoded ke dataset asli
encoded_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
df = df.join(encoded_df)


# In[10]:


# Menghapus kolom kategorikal asli yang telah di-OneHotEncode
df.drop(columns=categorical_columns, inplace=True)


# In[11]:


# 5. Normalisasi kolom numerik
scaler = StandardScaler()
numerical_columns = ['cap-diameter', 'stem-height', 'stem-width']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[12]:


# 6. Memisahkan fitur dan target
X = df.drop(columns=['class'])  # Semua kolom kecuali 'class' adalah fitur
y = df['class']  # Kolom 'class' adalah target


# In[13]:


# 7. Membagi dataset menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# 8. Training Model menggunakan RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[15]:


# 9. Evaluasi model
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[16]:


print(df.describe())
print(df['class'].value_counts())


# In[17]:


print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# In[ ]:




