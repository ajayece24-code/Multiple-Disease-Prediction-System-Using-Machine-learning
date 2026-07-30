from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "heart_model.pkl")
print("Model saved successfully")
