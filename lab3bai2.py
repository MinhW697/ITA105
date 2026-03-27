import pandas as pd
import matplotlib.pyplot as plt
#  BÀI 2: HEALTH 
health = pd.read_excel("ITA105_Lab_3_Health.xlsx")
health.columns = health.columns.str.lower().str.strip()

print("\n=== HEALTH ===")
print(health.isnull().sum())
print(health.describe())

health.hist(figsize=(10,6))
plt.suptitle("Health - Before")
plt.show()

health.plot(kind='box')
plt.title("Health Boxplot")
plt.show()

# Min-Max
health_minmax = (health - health.min()) / (health.max() - health.min())

# Z-score
health_z = (health - health.mean()) / health.std()

health_minmax.hist(figsize=(10,6))
plt.suptitle("Health Min-Max")
plt.show()

health_z.hist(figsize=(10,6))
plt.suptitle("Health Z-score")
plt.show()