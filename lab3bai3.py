import pandas as pd
import matplotlib.pyplot as plt
#  BÀI 3: FINANCE 
finance = pd.read_excel("ITA105_Lab_3_Finance.xlsx")
finance.columns = finance.columns.str.lower().str.strip()

print("\n=== FINANCE ===")
print(finance.describe())

finance.plot(kind='box')
plt.title("Finance Boxplot")
plt.show()

# Min-Max
finance_minmax = (finance - finance.min()) / (finance.max() - finance.min())

# Z-score
finance_z = (finance - finance.mean()) / finance.std()

# Scatter trước
plt.scatter(finance.iloc[:,0], finance.iloc[:,1])
plt.title("Finance Before")
plt.show()

# Scatter sau Min-Max
plt.scatter(finance_minmax.iloc[:,0], finance_minmax.iloc[:,1])
plt.title("Finance Min-Max")
plt.show()

# Scatter sau Z-score
plt.scatter(finance_z.iloc[:,0], finance_z.iloc[:,1])
plt.title("Finance Z-score")
plt.show()