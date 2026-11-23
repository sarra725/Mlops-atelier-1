import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def prepare_data(data_path="Churn_Modelling.csv"):
    """Charge le CSV, prétraite les données et retourne train/test normalisés."""
    encoder = LabelEncoder()
    data = pd.read_csv(data_path)

    # Colonnes inutiles pour la prédiction
    data = data.drop(["Surname", "Geography"], axis=1)

    # Encodage de Gender (Male/Female -> 1/0)
    data["Gender"] = encoder.fit_transform(data["Gender"])

    # Suppression des lignes avec valeurs manquantes
    data = data.dropna()

    # Caractéristiques (X) et cible (y)
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]

    # Split 80 % / 20 %
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Normalisation
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, "scaler.joblib")

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train):
    """Entraîne un RandomForest (100 arbres) et retourne le modèle."""
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle sur le jeu de test et retourne l'accuracy."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy


def save_model(model, model_path="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.joblib"):
    """Charge un modèle sauvegardé et le retourne."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict(features, model_path="model.joblib"):
    """Retourne les prédictions du modèle sauvegardé pour les features données."""
    model = load_model(model_path)
    return model.predict(features)
