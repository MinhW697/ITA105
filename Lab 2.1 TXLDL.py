#Câu 1
#1
import pandas as pd 
df = pd.read_csv("ITA105_Lab_2_Housing.csv")
print(df.shape)
print(df.isnull().sum())
#2
print(df.describe())
#3
import matplotlib.pyplot as plt 
import seaborn as sns
numeric_cols = ['dien_tich', 'gia', 'so_phong']
plt.figure(figsize=(15,5))
for i,col in enumerate(numeric_cols):
    plt.subplot(1,3,i+1)
    sns.boxplot(y=df[col], color= 'skyblue', flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 8})
    plt.title(f'Boxplot of {col}')
    plt.ylabel('Giá trị')
plt.tight_layout()
plt.show()

#4
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='dien_tich', y='gia', color='blue', alpha=0.6, label='Dữ liệu nhà ở')
outliers= df(df['dien_tich']> 800 | df['gia']>10000)
plt.scatter(outliers['dien_tich'], outliers['gia'], color='red', s=100, label='Điểm lạc lõng (Outliers)')
plt.title('Biểu đồ phân tán: Diện tích vs Giá nhà', fontsize=14)
plt.xlabel('Diện tích ($m^2$)')
plt.ylabel('Giá (Triệu VNĐ)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

#5
data=df['gia']

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Ngưỡng dưới: {lower_bound:.2f}")
print(f"Ngưỡng trên: {upper_bound:.2f}")

outliers = df[(df['gia'] < lower_bound) | (df['gia'] > upper_bound)]
print("\nCác dòng chứa điểm ngoại lệ:")
print(outliers)

#6
from scipy import stats
numeric_cols = ['dien_tich', 'gia', 'so_phong']
z_scores = stats.zscore(df[numeric_cols])
z_df = pd.DataFrame(z_scores, columns=numeric_cols)
outliers_mask = (z_df.abs() > 3).any(axis=1)
outliers_data = df[outliers_mask]
print("Các dòng được xác định là ngoại lệ theo Z-score:")
print(outliers_data)
print("\nGiá trị Z-score tương ứng của các điểm ngoại lệ:")
print(z_df[outliers_mask])

#7
import numpy as np 
column = 'gia'
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
upper_iqr = Q3 + 1.5 * IQR
lower_iqr = Q1 - 1.5 * IQR
outliers_iqr = df[(df[column] > upper_iqr) | (df[column] < lower_iqr)]
z = np.abs(stats.zscore(df[column]))
outliers_z = df[z > 3]
print(f"Số lượng ngoại lệ phát hiện bởi IQR (Boxplot): {len(outliers_iqr)}")
print(f"Số lượng ngoại lệ phát hiện bởi Z-score: {len(outliers_z)}")

#8
#Xác nhận ngoại lệ: Dữ liệu tồn tại các điểm ngoại lệ cực đoan (như dòng có diện tích 1000m², giá 50.000, 20 phòng ngủ). Các phương pháp IQR, Boxplot và Z-score ($|Z| > 3$) đều đồng nhất phát hiện ra các điểm này.
# Nguyên nhân: Đây là lỗi nhập liệu (Data Entry Error) vì các chỉ số vượt quá giới hạn logic thực tế của phân khúc nhà ở dân dụng và làm sai lệch nghiêm trọng phân phối chuẩn của tập dữ liệu.

#9
df_cleaned = df[df['gia'] < 10000].copy()

numeric_cols = ['dien_tich', 'gia', 'so_phong']

for col in numeric_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

print(f"Số lượng dòng sau khi làm sạch: {len(df_cleaned)}")
print(df_cleaned.describe())

#10
df_cleaned = df[df['gia'] < 10000].copy() 

for col in ['dien_tich', 'gia', 'so_phong']:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned[col] = df_cleaned[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('So sánh Boxplot TRƯỚC và SAU khi xử lý ngoại lệ', fontsize=16)

cols = ['dien_tich', 'gia', 'so_phong']

for i, col in enumerate(cols):
   
    sns.boxplot(y=df[col], ax=axes[0, i], color='salmon')
    axes[0, i].set_title(f'{col} (Gốc)')
    
    
    sns.boxplot(y=df_cleaned[col], ax=axes[1, i], color='limegreen')
    axes[1, i].set_title(f'{col} (Đã xử lý)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#**Kết luận Housing:** sau khi loại bỏ 3 bản ghi lỗi rõ ràng, phân phối dữ liệu ổn định hơn đáng kể; mean tiến gần median hơn, boxplot gọn hơn, nhưng vẫn giữ được một số ngôi nhà lớn/đắt có thể là ngoại lệ hợp lý ngoài thực tế.