import pandas as pd
df = pd.read_csv("ITA105_Lab_1.csv")
print(df.head())

#bai1
print("Kích thước dữ liệu",df.shape)
print(df.describe())
print(df.isnull().sum())

#bai2
df.isnull().sum()
df["Price"].fillna(df["Price"].mean(), inplace=True)
df["StockQuantity"].fillna(df["StockQuantity"].mean(), inplace=True)
df["Category"].fillna(df["Category"].mode(), inplace=True) 
df_drop=df.dropna() 
print("Kích thước sau khi drop",df_drop)
print("Kích thước sau khi fill", df.shape)

#bai3
print(df[["Price","StockQuantity"]].describe())
df = df[df["StockQuantity"] >= 0]
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

#bai4
df["Price_smooth"] = df["Price"].rolling(window=5).mean()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(df["Price"], label="Price gốc")
plt.plot(df["Price_smooth"], label="Price sau khi làm mượt")
plt.legend()
plt.title("So sánh Price trước và sau làm mượt")
plt.show()

#bai5
df["Category"] = df["Category"].str.lower()
df["Description"] = df["Description"].str.replace(r"[^\w\s]", "", regex=True)
df["Price_VND"] = df["Price"] * 24000
df.to_excel("data_clean.xlsx", index=False)