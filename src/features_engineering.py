import pandas as pd
import os

# Base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins dynamiques
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DATA_PATH = os.path.join(BASE_DIR, "data", "features")

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features temporelles au DataFrame pour une prévision trimestrielle.

    Args:
        df (pd.DataFrame): Le DataFrame initial contenant une colonne 'date'.

    Returns:
        pd.DataFrame: Le DataFrame avec les nouvelles features temporelles ajoutées.
    """
    if 'date' not in df.columns:
        raise ValueError("La colonne 'date' n'existe pas dans le DataFrame.")

    # Conversion de la colonne 'date' en datetime
    df['date'] = pd.to_datetime(df['date'])

    # Ajouter une colonne pour les trimestres
    df['quarter'] = df['date'].dt.to_period('Q')  # Année et trimestre (ex: 2023Q1)
    df['year'] = df['date'].dt.year  # Année

    return df

def encode_categorical_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Encode les variables catégoriques en utilisant le One-Hot Encoding.

    Args:
        df (pd.DataFrame): Le DataFrame initial contenant des colonnes catégoriques.
        categorical_columns (list): Liste des colonnes catégoriques à encoder.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes catégoriques encodées.
    """
    for col in categorical_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    return df_encoded

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les colonnes et catégories spécifiques pour les exclure du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame avec les features initiales.

    Returns:
        pd.DataFrame: Le DataFrame avec les features filtrées.
    """
    # Supprimer la colonne 'tech_event'
    if 'tech_event' in df.columns:
        df = df.drop(columns=['tech_event'])
        print("Colonne 'tech_event' supprimée.")

    # Exclusion de catégories spécifiques dans 'public_transport' et 'weather_condition'
    exclude_columns = [
        'weather_condition_Moderate',  # Exemple : Catégorie peu corrélée
        'public_transport_Limited'    # Exemple : Catégorie peu corrélée
    ]

    # Supprimer les colonnes si elles existent après encodage
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Colonne '{col}' supprimée.")

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'ensemble des étapes d'ingénierie des features au DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame initial.

    Returns:
        pd.DataFrame: Le DataFrame avec toutes les nouvelles features ajoutées.
    """
    # Ajout des features temporelles pour une prévision trimestrielle
    df = add_temporal_features(df)

    # Colonnes catégoriques à encoder
    categorical_columns = ['weather_condition', '5g_phase', 'public_transport']

    # Encodage des variables catégoriques
    df = encode_categorical_features(df, categorical_columns)

    # Filtrage des features non pertinentes
    df = filter_features(df)

    return df

if __name__ == "__main__":
    # Exemple de chargement d'un DataFrame pour tester
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
        output_path = os.path.join(FEATURES_DATA_PATH, "telecom_sales_data_features")
        df_features.to_csv(output_path, index=False)
        print(f"Features ajoutées et données sauvegardées dans {output_path}.")
    except Exception as e:
        print(f"Erreur lors de l'ingénierie des features : {e}")
