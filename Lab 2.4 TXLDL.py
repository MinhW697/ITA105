import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

def detect_multivariate_outliers(df, columns, contamination=0.05):
  
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
   
    outliers = iso_forest.fit_predict(df[columns].dropna())
    return outliers == -1


df_housing = pd.read_csv('ITA105_Lab_2_Housing.csv')
df_housing['is_outlier'] = detect_multivariate_outliers(df_housing, ['dien_tich', 'gia'])


df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv')
df_iot['is_outlier'] = detect_multivariate_outliers(df_iot, ['temperature', 'pressure'])

df_ecom = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')

df_ecom_clean = df_ecom[(df_ecom['price'] > 0) & (df_ecom['quantity'] > 0)].copy()
df_ecom_clean['is_outlier'] = detect_multivariate_outliers(df_ecom_clean, ['price', 'quantity', 'rating'])

print(f"Housing outliers: {df_housing['is_outlier'].sum()}")
print(f"IoT outliers: {df_iot['is_outlier'].sum()}")
print(f"Ecommerce outliers: {df_ecom_clean['is_outlier'].sum()}")

#2
def detect_outliers_summary(df, column_name):
 
    z_scores = np.abs(stats.zscore(df[column_name].dropna()))
    outliers_z = (z_scores > 3).sum()
    

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = ((df[column_name] < lower_bound) | (df[column_name] > upper_bound)).sum()
    
    return outliers_z, outliers_iqr


datasets = {
    "Housing": (pd.read_csv('ITA105_Lab_2_Housing.csv'), 'gia'),
    "IoT": (pd.read_csv('ITA105_Lab_2_Iot.csv'), 'temperature'),
    "Ecommerce": (pd.read_csv('ITA105_Lab_2_Ecommerce.csv'), 'price')
}

print(f"{'Dataset':<15} | {'Cột':<12} | {'Z-score (>3)':<15} | {'IQR (1.5x)':<15}")
print("-" * 65)

for name, (df, col) in datasets.items():
    z_count, iqr_count = detect_outliers_summary(df, col)
    print(f"{name:<15} | {col:<12} | {z_count:<15} | {iqr_count:<15}")

#3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
def mark_outliers(df, cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    df = df.copy()
  
    df = df.dropna(subset=cols)
    df['is_outlier'] = iso.fit_predict(df[cols])
    df['is_outlier'] = df['is_outlier'].map({1: 'Bình thường', -1: 'Ngoại lệ'})
    return df


df_h = pd.read_csv('ITA105_Lab_2_Housing.csv')
df_h_res = mark_outliers(df_h, ['dien_tich', 'gia'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_h_res, x='dien_tich', y='gia', hue='is_outlier', palette={'Bình thường': 'blue', 'Ngoại lệ': 'red'}, s=100)
plt.title('Housing: Diện tích vs Giá (Phát hiện ngoại lệ đa biến)')
plt.show()


df_i = pd.read_csv('ITA105_Lab_2_Iot.csv')
df_i_res = mark_outliers(df_i, ['temperature', 'pressure'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_i_res, x='temperature', y='pressure', hue='is_outlier', palette={'Bình thường': 'green', 'Ngoại lệ': 'red'}, alpha=0.6)
plt.title('IoT: Temperature vs Pressure (Phát hiện ngoại lệ đa biến)')
plt.show()


df_e = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')

df_e = df_e[df_e['price'] > 0]
df_e_res = mark_outliers(df_e, ['price', 'quantity', 'rating'])

sns.pairplot(df_e_res, vars=['price', 'quantity', 'rating'], hue='is_outlier', palette={'Bình thường': 'gray', 'Ngoại lệ': 'red'}, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Ecommerce: Scatter Matrix giữa các biến số', y=1.02)
plt.show()