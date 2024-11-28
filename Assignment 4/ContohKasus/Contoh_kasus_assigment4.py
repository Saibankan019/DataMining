#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np


# In[2]:


# 1. Buat DataFrame dari data
data = {
    "Tahun": list(range(1970, 2023)),
    "Produksi Padi(Ton)": [
        18693649, 20483687, 19393933, 21490578, 22476073, 22339455, 23300939, 23347132, 25771570,
        26282663, 29651905, 32774176, 33583677, 35303106, 38136446, 39032945, 39726761, 40078195,
        41676170, 44725582, 45178751, 44688247, 48240009, 48181087, 46641524, 49744140, 51101506,
        49377054, 49236692, 50866387, 51898852, 50460782, 51489694, 52137604, 54088468, 54151097,
        54454937, 57157435, 60325925, 64398890, 66469394, 65756904, 69056126, 71279709, 70846465,
        75397841, 79354767, 81148617, 59101577.84, 54604033.34, 54649202.24, 53802637.44, 54338410.44
    ]
}

df = pd.DataFrame(data)


# In[3]:


# 2. Preprocessing: Tambahkan fitur lagging
df["Lag_1"] = df["Produksi Padi(Ton)"].shift(1)


# In[4]:


# Hapus baris dengan nilai NaN akibat lagging
df = df.dropna()


# In[5]:


# 3. Siapkan fitur (X) dan target (y)
X = df[["Lag_1"]]  # Fitur: Produksi Padi sebelumnya
y = df["Produksi Padi(Ton)"]  # Target: Produksi Padi saat ini


# In[6]:


# 4. Bagi data menjadi training (70%) dan testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# In[7]:


# 5. Buat model Linear Regression dan latih
model = LinearRegression()
model.fit(X_train, y_train)


# In[8]:


# 6. Evaluasi model pada data testing
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Evaluasi Model:")
print(f"- MSE: {mse}")
print(f"- RMSE: {rmse}")
print(f"- MAPE: {mape * 100:.2f}%")


# In[9]:


# 7. Prediksi untuk tahun 2023 dan 2024
last_production = df["Produksi Padi(Ton)"].iloc[-1]
pred_2023 = model.predict([[last_production]])
pred_2024 = model.predict([[pred_2023[0]]])

print("\nPrediksi Produksi Padi:")
print(f"2023: {pred_2023[0]:,.2f} Ton")
print(f"2024: {pred_2024[0]:,.2f} Ton")

