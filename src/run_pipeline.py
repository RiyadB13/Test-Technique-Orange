import os
import subprocess

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to the scripts
DATA_PROCESSING_SCRIPT = os.path.join(BASE_DIR, "scripts", "data_processing.py")
FEATURE_ENGINEERING_SCRIPT = os.path.join(BASE_DIR, "scripts", "feature_engineering.py")
MODEL_SCRIPT = os.path.join(BASE_DIR, "scripts", "model.py")


def run_script(script_path):
    print(f"Execution du script : {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Succès : {script_path}\n{result.stdout}")
    else:
        print(f"Échec : {script_path}\n{result.stderr}")
        raise RuntimeError(f"Le script {script_path} a échoué.")


if __name__ == "__main__":
    try:
        print("=== Début du pipeline ===\n")

        # Step 1: Run data processing
        run_script(DATA_PROCESSING_SCRIPT)

        # Step 2: Run feature engineering
        run_script(FEATURE_ENGINEERING_SCRIPT)

        # Step 3: Run model training and prediction
        run_script(MODEL_SCRIPT)

        print("\n=== Pipeline terminé avec succès ===")
    except Exception as e:
        print(f"\nErreur pendant l'exécution du pipeline : {e}")
