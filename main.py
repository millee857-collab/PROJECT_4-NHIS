import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Processed_Flipdata.csv")
df.columns = df.columns.str.strip()

df['Prize'] = df['Prize'].astype(str).str.replace(',', '', regex=False)
df['Prize'] = pd.to_numeric(df['Prize'], errors='coerce')

df['Rear Camera'] = df['Rear Camera'].astype(str).str.replace('MP','', regex=False)
df['Rear Camera'] = pd.to_numeric(df['Rear Camera'], errors='coerce')

df['Front Camera'] = df['Front Camera'].astype(str).str.replace('MP','', regex=False)
df['Front Camera'] = pd.to_numeric(df['Front Camera'], errors='coerce')

for col in ['Model','Indexer','Keyarr','Axis Name']:
    if col in df.columns:
        df = df.drop(col, axis=1)

df = df.dropna()

cols = ['Prize','RAM','Battery_','Rear Camera','Front Camera']
cols = [c for c in cols if c in df.columns]

plt.figure(figsize=(8,6))
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.show()

plt.figure()
sns.histplot(df['Prize'], bins=30)
plt.show()

if 'RAM' in df.columns:
    plt.figure()
    sns.scatterplot(x=df['RAM'], y=df['Prize'])
    plt.show()

if 'Battery_' in df.columns:
    plt.figure()
    sns.scatterplot(x=df['Battery_'], y=df['Prize'])
    plt.show()

df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split

X = df.drop('Prize', axis=1)
y = df['Prize']

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

plt.figure()
feat_imp.head(10).plot(kind='barh')
plt.show()