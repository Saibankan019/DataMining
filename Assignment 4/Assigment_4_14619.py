#!/usr/bin/env python
# coding: utf-8

# ### Pandas: Untuk memuat dan memanipulasi data.
# #### Scikit-learn:
# ###### LinearRegression: Membuat model regresi linier.
# ###### Train-test split: Membagi data untuk pelatihan dan pengujian.
# ###### Metrics: Mengevaluasi performa model (MAE, MSE, R²).

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ## Simple Linear Regression:
# 
# ##### Input Data: data.csv dengan fitur horsepower sebagai variabel independen dan price sebagai target.
# ##### Langkah-langkah:
# ###### Data dicek untuk nilai kosong (null) dan diisi dengan rata-rata jika ada.
# ###### Data dibagi menjadi data latih (80%) dan data uji (20%).
# ###### Model dilatih untuk menghasilkan persamaan 
# ###### price = 𝑚 ⋅ horsepower + 𝑏
# ###### Prediksi diuji dan dievaluasi dengan metrik MAE, MSE, dan R².
# 
# ## Multivariable Linear Regression:
# 
# ##### Input Data: kc_house_data.csv dengan price sebagai target dan semua kolom numerik lainnya sebagai fitur (kecuali id dan date).
# ##### Langkah-langkah:
# ###### Kolom id dan date dihapus karena tidak relevan.
# ###### Data dicek untuk nilai kosong, dan diisi jika ada.
# ###### Jika terdapat fitur non-numerik, fitur tersebut diencoding menjadi numerik.
# ###### Data dibagi menjadi data latih (80%) dan data uji (20%).
# ###### Model dilatih untuk menghasilkan prediksi harga berdasarkan semua fitur numerik.
# ###### Evaluasi dilakukan menggunakan metrik yang sama.

# In[3]:


# --- Helper Function for Evaluation ---
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")
    return mae, mse, r2


# In[4]:


# --- Load and Preprocess Simple Linear Regression Dataset ---
data_simple = pd.read_csv('data.csv')
print("\nSimple Dataset Information:")
print(data_simple.info())


# In[5]:


# Check for missing values
if data_simple.isnull().any().sum() > 0:
    print("Missing values found. Filling missing values with column mean.")
    data_simple = data_simple.fillna(data_simple.mean())


# In[6]:


# Define X and y
X_simple = data_simple[['horsepower']].values
y_simple = data_simple['price'].values


# In[7]:


# Split data
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)


# In[8]:


# Train the model
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)


# In[9]:


# Predict and evaluate
y_pred_simple = model_simple.predict(X_test_simple)
print("\nSimple Linear Regression Equation:")
print(f"price = {model_simple.coef_[0]:.2f} * horsepower + {model_simple.intercept_:.2f}")
evaluate_model(y_test_simple, y_pred_simple, "Simple Linear Regression")


# In[10]:


# --- Load and Preprocess Multivariable Linear Regression Dataset ---
data_multivariable = pd.read_csv('kc_house_data.csv')
print("\nMultivariable Dataset Information:")
print(data_multivariable.info())


# In[11]:


# Drop unnecessary columns
data_multivariable_cleaned = data_multivariable.drop(['id', 'date'], axis=1)


# In[12]:


# Check for missing values
if data_multivariable_cleaned.isnull().any().sum() > 0:
    print("Missing values found. Filling missing values with column mean.")
    data_multivariable_cleaned = data_multivariable_cleaned.fillna(data_multivariable_cleaned.mean())


# In[13]:


# Define X and y
X_multivariable = data_multivariable_cleaned.drop('price', axis=1)
y_multivariable = data_multivariable_cleaned['price']


# In[14]:


# Check feature types and normalize if necessary
if np.any(X_multivariable.dtypes != 'float64') and np.any(X_multivariable.dtypes != 'int64'):
    print("Categorical data found. Encoding non-numeric columns.")
    X_multivariable = pd.get_dummies(X_multivariable, drop_first=True)


# In[15]:


# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multivariable, y_multivariable, test_size=0.2, random_state=42
)


# In[16]:


# Train the model
model_multivariable = LinearRegression()
model_multivariable.fit(X_train_multi, y_train_multi)


# ## Evaluasi Model:
# 
# #### Mean Absolute Error (MAE):
# ##### Rata-rata kesalahan absolut antara nilai aktual dan prediksi.
# #### Mean Squared Error (MSE):
# ##### Rata-rata kuadrat dari kesalahan prediksi.
# #### R² (Koefisien Determinasi):
# ##### Mengukur sejauh mana model menjelaskan variansi dalam data (0 hingga 1, semakin dekat ke 1 semakin baik).

# In[18]:


# Predict and evaluate
y_pred_multi = model_multivariable.predict(X_test_multi)
evaluate_model(y_test_multi, y_pred_multi, "Multivariable Linear Regression")

