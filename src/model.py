import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Définir les chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DATA_PATH = os.path.join(BASE_DIR, "data", "features")
MODELS_PATH = os.path.join(BASE_DIR, "models")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions")


# Fonction pour charger les données
def load_features_data(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(FEATURES_DATA_PATH, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    print(f"Chargement des données depuis {file_path}...")
    return pd.read_csv(file_path)


# Fonction pour entraîner le modèle
def train_model(X_train, y_train):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "y_pred": y_pred
    }


# Fonction pour sauvegarder les prévisions
def save_predictions(dates, y_test, y_pred, target):
    predictions_df = pd.DataFrame({
        "date": dates,
        "actual": y_test,
        "predicted": y_pred
    })
    predictions_file_path = os.path.join(PREDICTIONS_PATH, f"{target}_predictions.csv")
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    predictions_df.to_csv(predictions_file_path, index=False)
    print(f"Prévisions pour {target} sauvegardées dans {predictions_file_path}.")


# Fonction principale
if __name__ == "__main__":
    # Charger les données
    feature_file_name = "telecom_sales_data_features.csv"
    try:
        df = load_features_data(feature_file_name)
        print("Données chargées avec succès.")
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        exit(1)

    # Définir les features et les cibles
    features = ['marketing_score', 'competition_index', 'customer_satisfaction', 'purchasing_power_index',
                'store_traffic', 'year', 'month', 'day', 'day_of_week', 'week_of_year', 'is_weekend',
                'weather_condition_Bad', 'weather_condition_Good',
                '5g_phase_Early-5G', '5g_phase_Mature-5G', '5g_phase_Mid-5G', '5g_phase_Pre-5G',
                'public_transport_Good', 'public_transport_Hub', 'public_transport_Poor']
    targets = ['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']

    # Séparer les données en train et validation
    train_data = df[df['date'] < '2024-01-01']
    validation_data = df[(df['date'] >= '2024-01-01') & (df['date'] < '2025-01-01')]

    results = {}
    for target in targets:
        print(f"\nTraitement pour la cible : {target}")

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = validation_data[features]
        y_test = validation_data[target]
        dates_test = validation_data['date']  # Pour inclure les dates dans les prédictions

        # Entraîner le modèle
        best_model, best_params = train_model(X_train, y_train)
        print(f"Meilleurs paramètres pour {target} : {best_params}")

        # Évaluer le modèle
        evaluation_results = evaluate_model(best_model, X_test, y_test)
        print(f"Évaluation pour {target} : {evaluation_results}")

        # Stocker les résultats
        results[target] = {
            "model": best_model,
            "params": best_params,
            "evaluation": evaluation_results
        }

        # Sauvegarder le modèle
        model_path = os.path.join(MODELS_PATH, f"{target}_model.json")
        os.makedirs(MODELS_PATH, exist_ok=True)
        best_model.save_model(model_path)
        print(f"Modèle pour {target} sauvegardé dans {model_path}.")

        # Sauvegarder les prévisions
        save_predictions(dates_test, y_test, evaluation_results["y_pred"], target)

    print("\nTraitement terminé.")
