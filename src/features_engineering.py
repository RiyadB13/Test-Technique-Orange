import pandas as pd
import os

# Base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins dynamiques
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DATA_PATH = os.path.join(BASE_DIR, "data", "features")


def encode_categorical_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    for col in categorical_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    return df_encoded


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        raise ValueError("La colonne 'date' n'existe pas dans le DataFrame.")

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['week_of_year'] = df['date'].dt.isocalendar().week  # Numéro de la semaine
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 si c'est un week-end, sinon 0

    print("Features temporelles ajoutées : 'year', 'month', 'day', 'day_of_week', 'week_of_year', 'is_weekend'.")
    return df


def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    # Supprimer la colonne 'tech_event'
    if 'tech_event' in df.columns:
        df = df.drop(columns=['tech_event'])
        print("Colonne 'tech_event' supprimée.")

    # Supprimer la colonne 'city'
    if 'city' in df.columns:
        df = df.drop(columns=['city'])
        print("Colonne 'city' supprimée.")

    # Exclusion de catégories spécifiques dans 'public_transport' et 'weather_condition'
    exclude_columns = [
        'weather_condition_Moderate',  # Exemple : Catégorie peu corrélée
        'public_transport_Limited'  # Exemple : Catégorie peu corrélée
    ]

    # Supprimer les colonnes si elles existent après encodage
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Colonne '{col}' supprimée.")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ajouter les features temporelles
    df = add_temporal_features(df)

    # Colonnes catégoriques à encoder
    categorical_columns = ['weather_condition', '5g_phase', 'public_transport']

    # Encodage des variables catégoriques
    df = encode_categorical_features(df, categorical_columns)

    # Filtrage des features non pertinentes
    df = filter_features(df)

    return df


def save_features_data(df: pd.DataFrame, file_path: str):
    # Créer le répertoire s'il n'existe pas
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Vérifier si le fichier existe déjà
    if os.path.exists(file_path):
        print(f"Fichier existant trouvé : {file_path}. Il sera écrasé.")
    else:
        print(f"Le fichier n'existe pas, il sera créé : {file_path}.")

    # Sauvegarder les données
    df.to_csv(file_path, index=False)
    print(f"Features ajoutées et données sauvegardées dans {file_path}.")


if __name__ == "__main__":
    file_path = os.path.join(PROCESSED_DATA_PATH, "telecom_sales_data_filtered.csv")
    try:
        df = pd.read_csv(file_path)
        print("Chargement des données réussies.")
    except FileNotFoundError as e:
        print(f"Fichier introuvable : {file_path}")
        exit(1)

    # Application de l'ingénierie des features
    try:
        df_features = feature_engineering(df)
        output_path = os.path.join(FEATURES_DATA_PATH, "telecom_sales_data_features.csv")
        save_features_data(df_features, output_path)
    except Exception as e:
        print(f"Erreur lors de l'ingénierie des features : {e}")
