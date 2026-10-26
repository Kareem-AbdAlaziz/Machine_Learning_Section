import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Create a Simple Custom Dataset
data = {
    'Hours_Studied': [2, 8, 1, 9, 3, 7, 2, 10, 4, 6, 1.5, 8.5],
    'Sleep_Hours': [8, 6, 9, 5, 7, 6, 8, 4, 7, 5, 9, 6],
    'Passed': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df[['Hours_Studied', 'Sleep_Hours']]
y = df['Passed']

# 2. Split Data into Training and Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize the Random Forest Classifier
# n_estimators represents the number of decision trees in the forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the Model
rf_model.fit(X_train, y_train)

# 5. Make Predictions on the Test Set
y_pred = rf_model.predict(X_test)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Test with a New Example
# Let's predict for a student who studies 7 hours and gets 6 hours of sleep
new_student = [[7.0, 6.0]]
prediction = rf_model.predict(new_student)
result = "Pass" if prediction[0] == 1 else "Fail"
print(
    f"Prediction for a student who studies 7 hrs & sleeps 6 hrs: **{result}**"
)

# 8. Check Feature Importance
feature_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
)
print("\nFeature Importances:")
print(feature_importances)