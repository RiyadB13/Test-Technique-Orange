import pandas as pd
import os

# Base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins dynamiques
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


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
    Si le fichier existe déjà, il sera écrasé.

    Args:
        df (pd.DataFrame): Données transformées.
        file_name (str): Nom du fichier à enregistrer (CSV).
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

    if os.path.exists(file_path):
        print(f"Fichier existant trouvé : {file_path}. Il sera écrasé.")
    else:
        print(f"Le fichier n'existe pas, il sera créé : {file_path}.")

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
    if 'Unnamed: 0' in df.columns:
        df['date'] = df['Unnamed: 0']
        df = df.drop(columns=['Unnamed: 0'])

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    if 'date' in df.columns:
        df['month_period'] = df['date'].dt.to_period('M')

    numeric_columns = ['marketing_score', 'competition_index', 'purchasing_power_index',
                       'store_traffic', 'customer_satisfaction',
                       'jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']
    df[numeric_columns] = df[numeric_columns].interpolate(method='linear')

    categorical_columns = ['weather_condition', '5g_phase', 'public_transport']
    for col in categorical_columns:
        for period in df['month_period'].unique():
            mode_value = df[df['month_period'] == period][col].mode()
            if not mode_value.empty:
                df.loc[(df['month_period'] == period) & (df[col].isnull()), col] = mode_value[0]

    df = df.drop(columns=['month_period'])

    return df


def filter_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les données pour ne conserver qu'une ville choisie par l'utilisateur.

    Args:
        df (pd.DataFrame): DataFrame à filtrer.

    Returns:
        pd.DataFrame: DataFrame filtré pour la ville choisie.
    """
    if 'city' not in df.columns:
        raise ValueError("La colonne 'city' n'existe pas dans le DataFrame.")

    available_cities = df['city'].dropna().astype(str).unique()
    print("Villes disponibles :", ", ".join(available_cities))
    selected_city = input("Veuillez choisir une ville parmi les options ci-dessus : ").strip()

    if selected_city not in available_cities:
        raise ValueError(f"La ville '{selected_city}' n'existe pas dans les données.")

    df_filtered = df[df['city'] == selected_city]
    print(f"Nombre de lignes après filtrage pour {selected_city}: {len(df_filtered)}")

    return df_filtered


def add_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des lignes pour les dates du premier trimestre 2025 au DataFrame.

    Args:
        df (pd.DataFrame): DataFrame existant.

    Returns:
        pd.DataFrame: DataFrame étendu avec les futures dates ajoutées.
    """
    # Générer les futures dates pour Q1 2025
    future_dates = pd.date_range(start="2025-01-01", end="2025-03-31", freq="D")
    future_data = pd.DataFrame({'date': future_dates})

    # Ajouter des colonnes vides pour les futures dates
    numeric_columns = ['marketing_score', 'competition_index', 'purchasing_power_index',
                       'store_traffic', 'customer_satisfaction']
    categorical_columns = ['weather_condition', 'public_transport']

    for col in numeric_columns + categorical_columns + ['5g_phase']:
        future_data[col] = pd.NA

    # Ajouter les futures données au DataFrame
    df['date'] = pd.to_datetime(df['date'])
    future_data['date'] = pd.to_datetime(future_data['date'])
    df = pd.concat([df, future_data], ignore_index=True)

    # Interpolation des colonnes numériques pour chaque jour et mois
    for col in numeric_columns:
        df[col] = df.groupby([df['date'].dt.month, df['date'].dt.day])[col].transform(
            lambda x: x.interpolate(method='linear')
        )

    # Remplir les colonnes catégoriques (sauf 5g_phase) par le mode pour chaque jour du mois
    for col in categorical_columns:
        for month in df['date'].dt.month.unique():
            for day in df['date'].dt.day.unique():
                mode_value = df.loc[
                    (df['date'].dt.month == month) &
                    (df['date'].dt.day == day) &
                    (df['date'] < "2025-01-01"),  # Seulement les années précédentes
                    col
                ].mode()
                if not mode_value.empty:
                    df.loc[
                        (df['date'].dt.month == month) &
                        (df['date'].dt.day == day) &
                        (df['date'] >= "2025-01-01"),  # Dates futures
                        col
                    ] = mode_value[0]

    # Remplir 5g_phase par la valeur globale la plus fréquente
    if '5g_phase' in df.columns:
        mode_5g_phase = df.loc[df['date'] < "2025-01-01", '5g_phase'].mode()
        if not mode_5g_phase.empty:
            df['5g_phase'].fillna(mode_5g_phase[0], inplace=True)

    return df


if __name__ == "__main__":
    try:
        df_raw = load_raw_data("telecom_sales_data.csv")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    try:
        df_processed = process_data(df_raw)
    except ValueError as e:
        print(e)
        exit(1)

    try:
        df_filtered = filter_city(df_processed)
    except ValueError as e:
        print(e)
        exit(1)

    df_with_future = add_future_dates(df_filtered)

    save_processed_data(df_with_future, "telecom_sales_data_filtered.csv")
