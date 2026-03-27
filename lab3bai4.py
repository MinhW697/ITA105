import pandas as pd
import matplotlib.pyplot as plt
#  BÀI 4: GAMING 
gaming = pd.read_excel("ITA105_Lab_3_Gaming.xlsx")
gaming.columns = gaming.columns.str.lower().str.strip()

print("\n=== GAMING ===")
print(gaming.isnull().sum())

gaming.hist(figsize=(10,6))
plt.suptitle("Gaming Before")
plt.show()

# Min-Max
gaming_minmax = (gaming - gaming.min()) / (gaming.max() - gaming.min())

# Z-score
gaming_z = (gaming - gaming.mean()) / gaming.std()

gaming_minmax.hist(figsize=(10,6))
plt.suptitle("Gaming Min-Max")
plt.show()

gaming_z.hist(figsize=(10,6))
plt.suptitle("Gaming Z-score")
plt.show()