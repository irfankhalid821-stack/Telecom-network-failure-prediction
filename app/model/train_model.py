from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Reproducibility
np.random.seed(42)

# Simulated dataset
data = pd.DataFrame(
    {
        "signal_strength": np.random.randint(-100, -50, 1000),
        "temperature": np.random.randint(20, 50, 1000),
        "humidity": np.random.randint(20, 90, 1000),
        "network_load": np.random.randint(10, 100, 1000),
    }
)

# Target
# Failure likely with poor signal, high temperature, or very high load
data["failure"] = (
    (data["signal_strength"] < -85)
    | (data["temperature"] > 40)
    | (data["network_load"] > 80)
).astype(int)

X = data.drop("failure", axis=1)
y = data["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_path = Path(file).resolve().parent / "model.pkl"
joblib.dump(model, model_path)

print(f"Model trained and saved at: {model_path}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
