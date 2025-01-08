import pandas as pd
import os

# Chemins des fichiers
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"


def load_raw_data(file_name: str) -> pd.DataFrame:
    """
    Charge les données brutes depuis le dossier 'raw'.

    Args:
        file_name (str): Nom du fichier à charger (CSV).

    Returns:
        pd.DataFrame: Données brutes chargées.
    """
    file_path = os.path.join(RAW_DATA_PATH, file_name)
    if os.path.exists(file_path):
        print(f"Chargement des données depuis {file_path}...")
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Le fichier {file_name} n'existe pas dans {RAW_DATA_PATH}")


def save_processed_data(df: pd.DataFrame, file_name: str):
    """
    Enregistre les données traitées dans le dossier 'processed'.

    Args:
        df (pd.DataFrame): Données transformées.
        file_name (str): Nom du fichier à enregistrer (CSV).
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
    df.to_csv(file_path, index=False)
    print(f"Données enregistrées dans {file_path}.")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les transformations nécessaires au dataset.

    Args:
        df (pd.DataFrame): Données brutes.

    Returns:
        pd.DataFrame: Données transformées.
    """
    # Traitement des valeurs manquantes des colonnes numériques
    numeric_columns = ['marketing_score', 'competition_index', 'purchasing_power_index',
                       'store_traffic', 'customer_satisfaction']
    df[numeric_columns] = df[numeric_columns].interpolate(method='linear')

    # Traitement des valeurs manquantes des colonnes catégoriques par le mode mensuel
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        categorical_columns = ['weather_condition', '5g_phase', 'public_transport']
        for col in categorical_columns:
            for month in df['month'].unique():
                mode_value = df[df['month'] == month][col].mode()
                if not mode_value.empty:
                    df.loc[(df['month'] == month) & (df[col].isnull()), col] = mode_value[0]

    # Aucune imputation pour la colonne 'tech_event' car trop rare
    print("La colonne 'tech_event' n'a pas été traitée en raison de la rareté des données.")

    # Retourner le DataFrame transformé
    return df
