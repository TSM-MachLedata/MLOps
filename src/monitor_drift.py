import os
import json
import pandas as pd
from scipy.stats import ks_2samp
import mlflow
from datetime import datetime
import shutil


# ===============================
# 🔥 EXTRACTION AMÉLIORÉE DES FEATURES NUMÉRIQUES
# ===============================
def get_numeric_features(df, dataset_name):
    """Return a cleaned list of numeric features for drift detection."""

    # 1) Sélectionner colonnes numériques
    numeric_cols = [c for c in df.columns if df[c].dtype != "object"]

    # 2) Exclure colonnes inutiles
    exclude_cols = [
        "player", "player_name", "team", "team_name",
        "id", "match_id", "player_id"
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(numeric_cols) == 0:
        print(f"⚠️ Aucun feature numérique utile trouvé pour {dataset_name}")
        return []

    print(f"👉 {dataset_name} : {len(numeric_cols)} features analysées → {numeric_cols}")
    return numeric_cols


# ===============================
# 🔥 DRIFT DETECTION (KS TEST)
# ===============================
def detect_drift(current_df, reference_df, report_prefix, reports_path):
    """Generate a drift report for a pair of datasets."""

    # ===== PATHS REPORTS =====
    csv_report_path = os.path.join(reports_path, f"{report_prefix}_drift_report.csv")
    html_report_path = os.path.join(reports_path, f"{report_prefix}_drift_report.html")

    # ===== FEATURE SELECTION =====
    current_features = get_numeric_features(current_df, report_prefix)
    reference_features = get_numeric_features(reference_df, report_prefix)

    common_cols = list(set(current_features) & set(reference_features))

    if not common_cols:
        print(f"⚠️ Aucune colonne commune pour {report_prefix}")
        return None

    results = []

    # ===== DRIFT TEST =====
    for col in common_cols:
        ref_col = reference_df[col].dropna()
        cur_col = current_df[col].dropna()

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        stat, p_value = ks_2samp(ref_col, cur_col)
        drift = p_value < 0.05  # seuil KS

        results.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": drift
        })

    drift_df = pd.DataFrame(results)
    drift_df.to_csv(csv_report_path, index=False)

    drift_count = drift_df["drift_detected"].sum()
    total = len(drift_df)
    drift_rate = drift_count / total if total > 0 else 0

    print(f"📄 Rapport CSV enregistré : {csv_report_path}")
    print(f"📊 {drift_count}/{total} features en drift ({drift_rate:.1%})")

    # ===== HTML REPORT =====
    html_content = f"""
    <html>
    <head><title>Data Drift Report - {report_prefix}</title></head>
    <body>
        <h1>⚙️ Data Drift Report — {report_prefix}</h1>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Total features:</b> {total}</p>
        <p><b>Drift detected:</b> {drift_count} ({drift_rate:.1%})</p>
        <hr>
        {drift_df.to_html(index=False)}
    </body>
    </html>
    """

    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"📄 Rapport HTML enregistré : {html_report_path}")

    return drift_rate, csv_report_path, html_report_path



# ===============================
# 🔥 MAIN PIPELINE
# ===============================
def main():
    print("📊 Début du monitoring Data Drift...")

    processed_path = "data/processed"
    raw_path = "data/raw"
    reports_path = "reports"
    os.makedirs(reports_path, exist_ok=True)

    datasets = {
        "model1_clean": os.path.join(processed_path, "clean_matches.csv"),
        "player_strengths": os.path.join(processed_path, "player_strengths.csv"),
        "team_match_stats": os.path.join(raw_path, "team_match_stats_model2.csv"),
        "team_season_stats": os.path.join(raw_path, "team_season_stats_model2.csv"),
    }

    mlflow.set_experiment("football_prediction_mlops")

    drift_rates = {}

    with mlflow.start_run(run_name="data_drift_monitoring"):

        for name, current_path in datasets.items():

            print(f"\n🔍 Vérification du drift pour : {name}")

            if not os.path.exists(current_path):
                print(f"❌ Fichier manquant : {current_path}")
                continue

            current_df = pd.read_csv(current_path)
            reference_path = current_path.replace(".csv", "_reference.csv")

            # FIRST RUN → CREATE REFERENCE
            if not os.path.exists(reference_path):
                current_df.to_csv(reference_path, index=False)
                print(f"🆕 Référence créée : {reference_path}")
                continue

            reference_df = pd.read_csv(reference_path)

            result = detect_drift(current_df, reference_df, name, reports_path)
            if result is None:
                continue

            drift_rate, csv_report, html_report = result
            drift_rates[name] = drift_rate

            mlflow.log_metric(f"{name}_drift_rate", drift_rate)
            mlflow.log_artifact(csv_report)
            mlflow.log_artifact(html_report)

            # AUTO-UPDATE REFERENCE IF DRIFT > 30%
            THRESHOLD = 0.30
            if drift_rate > THRESHOLD:
                backup_path = reference_path.replace(
                    ".csv",
                    f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                shutil.copy(reference_path, backup_path)
                current_df.to_csv(reference_path, index=False)

                print(f"🔁 Mise à jour référence ({name}) car drift > 30%")
                print(f"📦 Ancienne référence sauvegardée : {backup_path}")
            else:
                print(f"✅ Pas de drift majeur pour {name}")

    # SAVE SUMMARY FOR retrain_if_drift.py
    summary_path = os.path.join(reports_path, "drift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(drift_rates, f, indent=2)

    print(f"\n📄 Résumé du drift enregistré dans {summary_path}")
    print("🎯 Monitoring terminé.")



if __name__ == "__main__":
    main()
