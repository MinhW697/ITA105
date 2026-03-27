#1
import pandas as pd
df_iot = pd.read_csv(r"D:\ITA105 - Tiền xử lý dữ liệu-20260311T030212Z-3-001\ITA105 - Tiền xử lý dữ liệu\Labs\Lab2\ITA105_Lab_2_Iot.csv")
df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
df_iot.set_index('timestamp', inplace=True)
print("\n--- Kiểm tra giá trị thiếu ---")
missing_values = df_iot.isnull().sum()
print(missing_values)

#2
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 6))
sns.lineplot(data=df_iot, x=df_iot.index, y='temperature', hue='sensor_id')

plt.title('Biến thiên nhiệt độ theo thời gian của các Sensor')
plt.xlabel('Thời gian')
plt.ylabel('Nhiệt độ (°C)')
plt.xticks(rotation=45) 
plt.legend(title='Cảm biến')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#3
window_size = 10

df_iot['rolling_mean'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(window=window_size).mean())
df_iot['rolling_std'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(window=window_size).std())

df_iot['upper_bond'] = df_iot['rolling_mean'] + 3 * df_iot['rolling_std']
df_iot['lower_bond'] = df_iot['rolling_mean'] - 3 * df_iot['rolling_std']

df_iot['is_outlier'] = (df_iot['temperature'] > df_iot['upper_bond']) | (df_iot['temperature'] < df_iot['lower_bond'])

outliers = df_iot[df_iot['is_outlier'] == True]
print(f"Số lượng ngoại lệ phát hiện được: {len(outliers)}")
print(outliers[['sensor_id', 'temperature', 'rolling_mean', 'is_outlier']].head())

plt.figure(figsize=(15, 6))
s1_data = df_iot[df_iot['sensor_id'] == 'S1']

plt.plot(s1_data.index, s1_data['temperature'], label='Nhiệt độ thực tế', alpha=0.5)
plt.plot(s1_data.index, s1_data['rolling_mean'], label='Rolling Mean (W=10)', color='orange')
plt.fill_between(s1_data.index, s1_data['lower_bond'], s1_data['upper_bond'], color='gray', alpha=0.2, label='Ngưỡng 3xStd')

plt.scatter(outliers[outliers['sensor_id'] == 'S1'].index, 
            outliers[outliers['sensor_id'] == 'S1']['temperature'], 
            color='red', label='Ngoại lệ', zorder=5)

plt.title('Phát hiện ngoại lệ bằng Rolling Statistics (Sensor S1)')
plt.legend()
plt.show()

#4
from scipy import stats
df_iot['z_score'] = df_iot.groupby('sensor_id')['temperature'].transform(lambda x: stats.zscore(x, ddof=0))
df_iot['is_outlier_z'] = df_iot['z_score'].abs() > 3

z_outliers = df_iot[df_iot['is_outlier_z']]

print(f"Tổng số điểm ngoại lệ phát hiện bằng Z-score (|Z| > 3): {len(z_outliers)}")
print("\n--- Một số điểm ngoại lệ tiêu biểu ---")
print(z_outliers[['timestamp', 'sensor_id', 'temperature', 'z_score']].head())
print("\n--- Số lượng ngoại lệ theo từng Sensor ---")
print(z_outliers['sensor_id'].value_counts())

#5
cols_to_check = ['temperature', 'pressure', 'humidity']

z_scores = stats.zscore(df_iot[cols_to_check])
is_outlier = (abs(z_scores) > 3).any(axis=1)
df_iot['is_outlier'] = is_outlier

plt.figure(figsize=(15, 5))
for i, col in enumerate(cols_to_check):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(data=df_iot, y=col, x='sensor_id', palette='Set2')
    plt.title(f'Boxplot of {col.capitalize()}')

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

sns.scatterplot(data=df_iot[~df_iot['is_outlier']], x='temperature', y='pressure', 
                alpha=0.4, label='Bình thường', ax=ax1)
sns.scatterplot(data=df_iot[df_iot['is_outlier']], x='temperature', y='pressure', 
                color='red', s=100, label='Ngoại lệ', ax=ax1)
ax1.set_title('Mối tương quan: Temperature vs Pressure')

sns.scatterplot(data=df_iot[~df_iot['is_outlier']], x='pressure', y='humidity', 
                alpha=0.4, label='Bình thường', ax=ax2)
sns.scatterplot(data=df_iot[df_iot['is_outlier']], x='pressure', y='humidity', 
                color='red', s=100, label='Ngoại lệ', ax=ax2)
ax2.set_title('Mối tương quan: Pressure vs Humidity')

plt.legend()
plt.show()

#6
count_z = df_iot['is_outlier_z'].sum()
count_rolling = df_iot['is_outlier_rolling'].sum()

Q1 = df_iot['temperature'].quantile(0.25)
Q3 = df_iot['temperature'].quantile(0.75)
IQR = Q3 - Q1
count_boxplot = ((df_iot['temperature'] < (Q1 - 1.5 * IQR)) | (df_iot['temperature'] > (Q3 + 1.5 * IQR))).sum()

print(f"Số lượng ngoại lệ (Boxplot - IQR): {count_boxplot}")
print(f"Số lượng ngoại lệ (Z-score > 3): {count_z}")
print(f"Số lượng ngoại lệ (Rolling Mean 3-sigma): {count_rolling}")

#7
df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
df_iot.set_index('timestamp', inplace=True)
df_cleaned = df_iot.copy()

cols = ['temperature', 'pressure', 'humidity']

for sensor in df_cleaned['sensor_id'].unique():
    mask = df_cleaned['sensor_id'] == sensor
    
    for col in cols:
        
        Q1 = df_cleaned.loc[mask, col].quantile(0.25)
        Q3 = df_cleaned.loc[mask, col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 3 * IQR 
        upper_limit = Q3 + 3 * IQR
        
      
        df_cleaned.loc[mask & ((df_cleaned[col] < lower_limit) | (df_cleaned[col] > upper_limit)), col] = np.nan
        
        
        df_cleaned.loc[mask, col] = df_cleaned.loc[mask, col].interpolate(method='linear')
        
        
        low_clip = Q1 - 1.5 * IQR
        up_clip = Q3 + 1.5 * IQR
        df_cleaned.loc[mask, col] = df_cleaned.loc[mask, col].clip(lower=low_clip, upper=up_clip)

plt.figure(figsize=(15, 8))


plt.subplot(2, 1, 1)
sns.lineplot(data=df_iot, x=df_iot.index, y='temperature', hue='sensor_id', alpha=0.5)
plt.title('Dữ liệu Temperature GỐC (Nhiều gai nhiễu)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
sns.lineplot(data=df_cleaned, x=df_cleaned.index, y='temperature', hue='sensor_id')
plt.title('Dữ liệu Temperature SAU XỬ LÝ (Interpolation & Clipping)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()