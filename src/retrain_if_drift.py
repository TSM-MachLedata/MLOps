import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import json

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.30"))
SUMMARY_PATH = Path("reports/drift_summary.json")


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def run_monitor_drift():
    log(" Lancement du monitoring de drift (monitor_drift.py)...")
    subprocess.check_call([sys.executable, "src/monitor_drift.py"])



def read_drift_summary():
    if not SUMMARY_PATH.exists():
        log(f" Fichier {SUMMARY_PATH} introuvable, aucun drift lu.")
        return {}

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(f" Drift summary : {data}")
    return data


def retrain_models():
    """
    Quand drift > seuil : on réentraîne tous les modèles + re-sélection du champion.
    """

    # MODEL 1 (multi-ligues)
    log(" Retrain MODEL 1 (stage DVC: train)")
    subprocess.check_call(["dvc", "repro", "train"])

    # MODEL 2 (XGB 3 classes)
    log(" Retrain MODEL 2 (stage DVC: train_model2)")
    subprocess.check_call(["dvc", "repro", "train_model2"])

    # MODEL 3 (forces joueurs)
    log(" Update MODEL 3 (stage DVC: build_player_strengths)")
    subprocess.check_call(["dvc", "repro", "build_player_strengths"])

    # Sélection du champion (lancera aussi eval_model3_player_mode si besoin)
    log(" Evaluate player-mode & select champion (stage DVC: select_champion)")
    subprocess.check_call(["dvc", "repro", "select_champion"])

    # Push vers GCS via DVC
    log(" dvc push (data + modèles + champion vers GCS)...")
    subprocess.check_call(["dvc", "push"])
    log(" dvc push terminé.")



def main():
    log(" Début pipeline monitoring + retrain conditionnel")

    # 1) Calcul du drift + MAJ des références
    run_monitor_drift()

    # 2) Lire les valeurs de drift
    drifts = read_drift_summary()
    if not drifts:
        log(" Aucun drift lu, on ne réentraîne pas.")
        return

    max_drift = max(drifts.values())
    log(f" Max drift détecté = {max_drift:.1%} (seuil = {DRIFT_THRESHOLD:.0%})")

    if max_drift <= DRIFT_THRESHOLD:
        log(" Drift sous le seuil, pas de ré-entrainement.")
        return

    log(" Drift > seuil → ré-entrainement des modèles.")
    retrain_models()


if __name__ == "__main__":
    main()
