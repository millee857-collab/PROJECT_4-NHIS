import pandas as pd

df = pd.read_csv("Processed_Flipdata.csv")

df.columns = df.columns.str.strip()

print(df.head())
print(df.info())
# Camera columns clean
df['Rear Camera'] = df['Rear Camera'].astype(str).str.replace('MP','').astype(float)
df['Front Camera'] = df['Front Camera'].astype(str).str.replace('MP','').astype(float)

# Drop model column
if 'Model' in df.columns:
    df = df.drop('Model', axis=1)

# Remove missing values
df = df.dropna()

print("Cleaning Done")

df = pd.get_dummies(df, drop_first=True)

print("Encoding Done")
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

cols = ['RAM', 'Battery_', 'Rear Camera', 'Front Camera']

plt.figure(figsize=(8,6))
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Important Features Heatmap")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.histplot(df['Prize'], bins=30)
plt.title("Prize Distribution")
plt.xlabel("Prize")
plt.ylabel("Count")
plt.show()