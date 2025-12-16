import os
import json

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
)
import mlflow
import mlflow.xgboost


def main():
    print(" Démarrage de l’entraînement des modèles XGBoost...")

    # Charger les données prétraitées
    data_path = "data/processed/clean_matches.csv"
    data = pd.read_csv(data_path)
    print(f" Données chargées : {data.shape[0]} matchs, {data.shape[1]} colonnes")

    # Sélection des features numériques pertinentes
    features = [
        "home_matches_played",
        "home_goals_for",
        "home_goals_against",
        "home_goals_diff",
        "away_matches_played",
        "away_goals_for",
        "away_goals_against",
        "away_goals_diff",
    ]

    for f in features:
        if f not in data.columns:
            raise ValueError(f"⚠️ La colonne '{f}' est manquante dans les données.")

    X = data[features]
    y_home = data["home_goals"]
    y_away = data["away_goals"]

    # Division train/test
    X_train, X_test, y_home_train, y_home_test = train_test_split(
        X, y_home, test_size=0.2, random_state=42
    )
    _, _, y_away_train, y_away_test = train_test_split(
        X, y_away, test_size=0.2, random_state=42
    )

    # Hyperparams utilisés pour les deux modèles
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 5

    # Configuration MLflow
    mlflow.set_experiment("football_prediction_mlops")
    with mlflow.start_run(run_name="xgboost_multi_leagues"):
        # Tags & params comme pour model2 / model3
        mlflow.set_tag("model_name", "model1")
        mlflow.set_tag("stage", "train")

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("features", features)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Entraînement des modèles
        model_home = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        model_away = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )

        model_home.fit(X_train, y_home_train)
        model_away.fit(X_train, y_away_train)

        # Évaluation régression
        y_home_pred = model_home.predict(X_test)
        y_away_pred = model_away.predict(X_test)

        metrics = {
            "mse_home": mean_squared_error(y_home_test, y_home_pred),
            "mae_home": mean_absolute_error(y_home_test, y_home_pred),
            "r2_home": r2_score(y_home_test, y_home_pred),
            "mse_away": mean_squared_error(y_away_test, y_away_pred),
            "mae_away": mean_absolute_error(y_away_test, y_away_pred),
            "r2_away": r2_score(y_away_test, y_away_pred),
        }

        # Métriques "classification" dérivées (Home/Away/Draw)
        def to_result(home_g, away_g):
            if home_g > away_g:
                return "Home Win"
            elif home_g < away_g:
                return "Away Win"
            else:
                return "Draw"

        y_true_res = [to_result(h, a) for h, a in zip(y_home_test, y_away_test)]
        y_pred_res = [to_result(h, a) for h, a in zip(y_home_pred, y_away_pred)]

        acc = accuracy_score(y_true_res, y_pred_res)
        f1 = f1_score(y_true_res, y_pred_res, average="macro")

        metrics["accuracy"] = acc
        metrics["f1_macro"] = f1

        # Log des métriques dans MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print("📊 Résultats du modèle :")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Sauvegarde JSON pour comparaison de modèles (model1_metrics.json)
        os.makedirs("reports", exist_ok=True)
        model1_metrics_path = os.path.join("reports", "model1_metrics.json")
        with open(model1_metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        # Sauvegarde des modèles
        os.makedirs("app/models", exist_ok=True)
        home_model_path = "app/models/home_model.json"
        away_model_path = "app/models/away_model.json"
        model_home._estimator_type = "regressor"
        model_away._estimator_type = "regressor"
        model_home.save_model(home_model_path)
        model_away.save_model(away_model_path)


        # log artifacts dans MLflow (modèles + dataset d'entraînement)
        mlflow.log_artifact(home_model_path)
        mlflow.log_artifact(away_model_path)
        if os.path.exists(data_path):
            mlflow.log_artifact(data_path)
        mlflow.log_artifact(model1_metrics_path)

        print(" Modèles sauvegardés et enregistrés dans MLflow.")


if __name__ == "__main__":
    main()
