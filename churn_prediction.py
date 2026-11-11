# Bank Customer Churn Prediction
# Author: Maroua Bachiri
# Tools: Python, Pandas, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('Churn_Modelling.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Data Cleaning - Drop useless columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. One-Hot Encoding for Geography and Gender
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# 4. Split Features and Target
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.1f}%")
print(classification_report(y_test, y_pred_rf))

# 6. Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.1f}%")

# 7. Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop Features:")
print(importances.sort_values(ascending=False).head(5))

# 8. Visualization
importances.sort_values(ascending=False).head(5).plot(kind='bar')
plt.title('Top 5 Features Driving Churn')
plt.tight_layout()
plt.show()
