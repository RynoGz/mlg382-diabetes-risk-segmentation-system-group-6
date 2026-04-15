"""
Final model training script for Diabetes Risk Classification System.
Trains the selected Random Forest model and saves it for deployment.
"""

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# 1. Load Data
# ----------------------------
X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv")
y_test = pd.read_csv("../data/y_test.csv")

# Flatten target
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# ----------------------------
# 2. Train Final Model
# ----------------------------
final_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None 
)

final_model.fit(X_train, y_train)

# ----------------------------
# 3. Predictions
# ----------------------------
y_pred = final_model.predict(X_test)

# ----------------------------
# 4. Evaluation
# ----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 5. Save Model
# ----------------------------
joblib.dump(final_model, "../artifacts/classification_model.pkl")

print("\nModel saved successfully to ../artifacts/classification_model.pkl")