import pandas as pd
import matplotlib.pyplot as plt
#  BÀI 1: SPORTS 
sports = pd.read_excel("ITA105_Lab_3_Sports.xlsx")
sports.columns = sports.columns.str.lower().str.strip()

print("\n=== SPORTS ===")
print(sports.isnull().sum())
print(sports.describe())

sports.hist(figsize=(10,6))
plt.suptitle("Sports - Before")
plt.show()

sports.plot(kind='box')
plt.title("Sports Boxplot")
plt.show()

# Min-Max
sports_minmax = (sports - sports.min()) / (sports.max() - sports.min())

# Z-score
sports_z = (sports - sports.mean()) / sports.std()

sports_minmax.hist(figsize=(10,6))
plt.suptitle("Sports Min-Max")
plt.show()

sports_z.hist(figsize=(10,6))
plt.suptitle("Sports Z-score")
plt.show()