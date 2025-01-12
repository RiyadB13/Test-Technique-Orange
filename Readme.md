# Optimisation des Revenus des Smartphones - DataTel

Ce projet vise à analyser les données historiques de ventes de smartphones, développer des modèles de prévision robustes pour anticiper les revenus, et fournir des recommandations stratégiques basées sur les résultats. Il a été réalisé pour l'opérateur télécom **DataTel** afin d'optimiser la stratégie de vente de ses modèles phares.

## Structure du projet

Le projet se compose des fichiers et répertoires suivants :

### Résultat en markdown
```
├── data/
│   ├── raw/               # Contient les données brutes
│   ├── processed/         # Contient les données prétraitées
│   └── features/          # Contient les données avec features ajoutées
├── scripts/
│   ├── data_processing.py     # Script pour prétraiter les données brutes (non utilisé ici)
│   ├── feature_engineering.py # Script pour l'ingénierie des features
│   ├── model.py               # Script pour entraîner le modèle et faire des prédictions
├── models/
│   ├── jPhone_Pro_revenue_model.json
│   ├── Kaggle_Pixel_5_revenue_model.json
│   ├── Planet_SX_revenue_model.json
├── requirements.txt           # Liste des dépendances Python
├── README.md                  # Ce fichier
└── presentation/
    └── results.pdf            # Présentation des résultats
```
---

## Prérequis

Assurez-vous d'avoir **Python 3.8+** installé sur votre système.

### Installation des dépendances

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```
### Lancement du pipeline
Après avoir installé les dépendances, exécutez le code grâce au script **run_pipeline.py**.
```bash
cd src
python run_pipeline.py
```

Dans ce projet, j'ai créé un modèle qui prédit les revenus du trimestre 2025 d'une ville au choix. 
Le rapport a été réalisé pour les prédictions de la ville de Marseille.