import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_PATH = "bank_model.pkl"
SCALER_PATH = "scaler.pkl"

def load_data():
    df = pd.read_csv("bank_marketing_data_with_y.csv", sep=',')

    # Check if 'y' column exists
    if "y" not in df.columns:
        raise ValueError("'y' column not found in dataset. Please ensure the CSV includes a 'y' column (target variable) with 'yes'/'no' values.")

    # Convert target to binary
    df['y'] = df['y'].map({"yes": 1, "no": 0})

    # Label encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop("y", axis=1)
    y = df["y"]

    return X, y

def train_and_save_model():
    X, y = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model and scaler saved.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()