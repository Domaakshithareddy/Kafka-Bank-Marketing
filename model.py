import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib
import os
import matplotlib.pyplot as plt

BACKEND_DIR = "backend"
SCALER_PATH = os.path.join(BACKEND_DIR, "scaler.pkl")
BANK_MODEL_PATH = os.path.join(BACKEND_DIR, "bank_model.pkl")
METRICS_PLOT_PATH = os.path.join(BACKEND_DIR, "model_comparison.png")

def load_data():
    df = pd.read_csv("bank-full.csv", sep=';')
    df['y'] = df['y'].map({"yes": 1, "no": 0})
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop("y", axis=1)
    y = df["y"]
    return X, y

def train_and_evaluate_models():
    os.makedirs(BACKEND_DIR, exist_ok=True)
    
    X, y = load_data()
    print("Class distribution (y):")
    print(pd.Series(y).value_counts(normalize=True))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=10, metric='manhattan'),
        'SVM': SVC(C=1.0, kernel='rbf', random_state=42, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(C=1.0, random_state=42, class_weight='balanced')
    }
    
    metrics = {'Accuracy': [], 'Precision': [], 'F1 Score': [], 'Recall': []}
    model_names = []
    best_model = None
    best_accuracy = 0
    best_model_name = ''
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['F1 Score'].append(f1)
        metrics['Recall'].append(recall)
        model_names.append(name)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
        
        print(f"{name} Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}\n")
    
    joblib.dump(scaler, SCALER_PATH)
    if best_model is not None:
        joblib.dump(best_model, BANK_MODEL_PATH)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2
    
    ax.bar(x - width*1.5, metrics['Accuracy'], width, label='Accuracy')
    ax.bar(x - width/2, metrics['Precision'], width, label='Precision')
    ax.bar(x + width/2, metrics['F1 Score'], width, label='F1 Score')
    ax.bar(x + width*1.5, metrics['Recall'], width, label='Recall')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(METRICS_PLOT_PATH)
    plt.close()
    
    print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    print(f"Scaler saved as {SCALER_PATH}")
    print(f"Best model saved as {BANK_MODEL_PATH}")
    print(f"Comparison plot saved as {METRICS_PLOT_PATH}")
    
    return best_model, scaler

if __name__ == "__main__":
    train_and_evaluate_models()