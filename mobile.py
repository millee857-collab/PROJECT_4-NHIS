import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Processed_Flipdata.csv")

df.columns = df.columns.str.strip()

if 'Rear Camera' in df.columns:
    df['Rear Camera'] = df['Rear Camera'].astype(str).str.replace('MP', '', regex=False).astype(float)

if 'Front Camera' in df.columns:
    df['Front Camera'] = df['Front Camera'].astype(str).str.replace('MP', '', regex=False).astype(float)

if 'Model' in df.columns:
    df = df.drop('Model', axis=1)

df = df.dropna()

df = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

if 'RAM' in df.columns and 'Price' in df.columns:
    sns.scatterplot(x=df['RAM'], y=df['Price'])
    plt.show()

sns.histplot(df['Price'], bins=30)
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

importance = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

print(feat_imp.head(10))

feat_imp.head(10).plot(kind='barh')
plt.show()