import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_data(data_path="Churn_Modelling.csv"):
    """Importe les données depuis le fichier CSV."""
    data = pd.read_csv(data_path)
    return data


def clean_data(data):
    """Supprime les colonnes inutiles et les lignes vides."""
    data = data.drop(["Surname", "Geography"], axis=1)
    data = data.dropna()
    return data


def encode_data(data):
    """Encode la colonne Gender en nombres."""
    encoder = LabelEncoder()
    data["Gender"] = encoder.fit_transform(data["Gender"])
    return data


def split_data(data):
    """Sépare X et y, puis train et test."""
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return x_train, x_test, y_train, y_test


def normalize_data(x_train, x_test):
    """Normalise les données avec StandardScaler."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")
    return x_train_scaled, x_test_scaled


def prepare_data(data_path="Churn_Modelling.csv"):
    """Exécute tout le prétraitement."""
    data = load_data(data_path)
    data = clean_data(data)
    data = encode_data(data)
    x_train, x_test, y_train, y_test = split_data(data)
    x_train, x_test = normalize_data(x_train, x_test)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Entraîne le modèle RandomForest."""
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle (accuracy)."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy


def save_model(model, model_path="model.joblib"):
    """Sauvegarde le modèle."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.joblib"):
    """Charge le modèle sauvegardé."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict(features, model_path="model.joblib", scaler_path="scaler.joblib"):
    """Prédit avec le modèle sauvegardé (features non normalisées)."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)
