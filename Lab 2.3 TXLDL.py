#1
import pandas as pd
df_ecom = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')

print("--- Kiểm tra giá trị thiếu (Missing Values) ---")
missing_info = df_ecom.isnull().sum()
print(missing_info)

print("\n--- Thống kê mô tả các biến số ---")
desc_stats = df_ecom.describe()
print(desc_stats)

#2
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Biểu đồ Boxplot cho các biến số Ecommerce', fontsize=16)

cols = ['price', 'quantity', 'rating']
colors = ['skyblue', 'lightgreen', 'salmon']

for i, col in enumerate(cols):
    sns.boxplot(y=df_ecom[col], ax=axes[i], color=colors[i])
    axes[i].set_title(f'Phân phối của {col.capitalize()}')
    axes[i].set_ylabel('Giá trị')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#3
import numpy as np
from scipy import stats
numeric_cols = ['price', 'quantity', 'rating']

z_scores = np.abs(stats.zscore(df_ecom[numeric_cols]))

outliers_z = (z_scores > 3).any(axis=1)

outliers_iqr = pd.Series([False] * len(df_ecom))

for col in numeric_cols:
    Q1 = df_ecom[col].quantile(0.25)
    Q3 = df_ecom[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
  
    col_outliers = (df_ecom[col] < lower_bound) | (df_ecom[col] > upper_bound)
    outliers_iqr = outliers_iqr | col_outliers


df_ecom['is_outlier_z'] = outliers_z
df_ecom['is_outlier_iqr'] = outliers_iqr

print(f"Số lượng ngoại lệ theo Z-score: {outliers_z.sum()}")
print(f"Số lượng ngoại lệ theo IQR: {outliers_iqr.sum()}")

print("\n--- Các dòng ngoại lệ tiêu biểu (IQR) ---")
print(df_ecom[outliers_iqr].head())

#4
z_price = np.abs(stats.zscore(df_ecom['price']))
z_qty = np.abs(stats.zscore(df_ecom['quantity']))

df_ecom['is_outlier'] = (z_price > 2) | (z_qty > 2)

plt.figure(figsize=(12, 8))

sns.scatterplot(data=df_ecom[~df_ecom['is_outlier']], 
                x='price', y='quantity', 
                hue='category', style='category', 
                s=60, alpha=0.6, label='Bình thường')

sns.scatterplot(data=df_ecom[df_ecom['is_outlier']], 
                x='price', y='quantity', 
                color='red', marker='X', s=150, 
                label='Ngoại lệ (Cực đoan)')

plt.title('Mối tương quan giữa Price và Quantity (Highlight Ngoại lệ)', fontsize=15)
plt.xlabel('Giá sản phẩm (Price)', fontsize=12)
plt.ylabel('Số lượng (Quantity)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#6
df_clean = df_ecom.copy()
initial_count = len(df_clean)
df_clean = df_clean[
    (df_clean['price'] > 0) & 
    (df_clean['quantity'] > 0) & 
    (df_clean['rating'] <= 5)
]
print(f"Đã loại bỏ {initial_count - len(df_clean)} dòng lỗi nhập liệu.")

price_upper_limit = df_clean['price'].quantile(0.99)
df_clean['price_clipped'] = df_clean['price'].clip(upper=price_upper_limit)

df_clean['quantity_log'] = np.log1p(df_clean['quantity'])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(df_ecom['price'], kde=True, ax=axes[0], color='red', label='Gốc')
sns.histplot(df_clean['price_clipped'], kde=True, ax=axes[0], color='green', label='Đã Clip')
axes[0].set_title("Phân phối Price (Clipped)")
axes[0].legend()

sns.histplot(df_ecom['quantity'], kde=True, ax=axes[1], color='red', label='Gốc')
sns.histplot(df_clean['quantity_log'], kde=True, ax=axes[1], color='blue', label='Log-transformed')
axes[1].set_title("Phân phối Quantity (Log-transform)")
axes[1].legend()

plt.tight_layout()
plt.show()
df_clean.to_csv("ITA105_Lab_2_Ecommerce_Cleaned.csv", index=False)

#7
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('So sánh Boxplot TRƯỚC và SAU khi xử lý (Ecommerce)', fontsize=16)

cols = ['price', 'quantity', 'rating']

for i, col in enumerate(cols):
   
    sns.boxplot(y=df_ecom[col], ax=axes[0, i], color='salmon')
    axes[0, i].set_title(f'{col} (Gốc)')
    
    
    target_col = col
    if col == 'price': target_col = 'price_clipped'
    if col == 'quantity': target_col = 'quantity_log'
    
    sns.boxplot(y=df_clean[target_col], ax=axes[1, i], color='limegreen')
    axes[1, i].set_title(f'{target_col} (Đã xử lý)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='price_clipped', y='quantity_log', hue='category', alpha=0.5)
plt.title('Scatter Plot sau khi xử lý (Clipped Price vs Log Quantity)')
plt.grid(True, alpha=0.3)
plt.show()