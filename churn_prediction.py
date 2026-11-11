# Bank Customer Churn Prediction
# Author: Maroua Bachiri
# Dataset: Churn_Modelling.csv (10,000 customers)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('Churn_Modelling.csv')
print(f"Dataset shape: {df.shape}")

# 2. Data Cleaning
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# 3. Split data
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Random Forest Model
rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.1f}%")
print(classification_report(y_test, y_pred_rf))

# 6. Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.1f}%")

# 7. Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 5 Features:")
print(importances.sort_values(ascending=False).head(5))
importances.sort_values(ascending=False).head(5).plot(kind='bar', color='teal')
plt.title('Top 5 Features Driving Churn')
plt.tight_layout()
plt.show()
