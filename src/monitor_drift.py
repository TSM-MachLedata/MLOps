import os
import json
import pandas as pd
from scipy.stats import ks_2samp
import mlflow
from datetime import datetime
import shutil


def detect_drift(current_df, reference_df, report_prefix, reports_path):
    """Generate a drift report for a pair of datasets."""

    csv_report_path = os.path.join(reports_path, f"{report_prefix}_drift_report.csv")
    html_report_path = os.path.join(reports_path, f"{report_prefix}_drift_report.html")

    # Colonnes numériques communes
    common_cols = list(
        set(current_df.select_dtypes(include="number").columns)
        & set(reference_df.select_dtypes(include="number").columns)
    )

    if not common_cols:
        print(f"⚠️ Aucune colonne numérique commune pour {report_prefix}")
        return None

    results = []
    for col in common_cols:
        ref_col = reference_df[col].dropna()
        cur_col = current_df[col].dropna()

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        stat, p_value = ks_2samp(ref_col, cur_col)
        drift = p_value < 0.05

        results.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": drift
        })

    drift_df = pd.DataFrame(results)
    drift_df.to_csv(csv_report_path, index=False)

    drift_count = drift_df["drift_detected"].sum()
    drift_rate = drift_count / len(drift_df) if len(drift_df) > 0 else 0

    print(f"📄 Rapport CSV enregistré : {csv_report_path}")
    print(f"📊 {drift_count}/{len(drift_df)} features en drift ({drift_rate:.1%})")

    # HTML report
    html_content = f"""
    <html>
    <head><title>Data Drift Report - {report_prefix}</title></head>
    <body>
        <h1>⚙️ Data Drift Report — {report_prefix}</h1>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Total features:</b> {len(drift_df)}</p>
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

    drift_rates = {}  # NEW: on garde le drift de chaque dataset

    with mlflow.start_run(run_name="data_drift_monitoring"):

        for report_prefix, current_path in datasets.items():

            print(f"\n🔍 Vérification du drift pour : {report_prefix}")

            if not os.path.exists(current_path):
                print(f"❌ Fichier manquant : {current_path}, skip...")
                continue

            current_df = pd.read_csv(current_path)
            reference_path = current_path.replace(".csv", "_reference.csv")

            if not os.path.exists(reference_path):
                current_df.to_csv(reference_path, index=False)
                print(f"🆕 Référence créée : {reference_path}")
                continue

            reference_df = pd.read_csv(reference_path)

            result = detect_drift(current_df, reference_df, report_prefix, reports_path)
            if result is None:
                continue

            drift_rate, csv_report, html_report = result

            drift_rates[report_prefix] = drift_rate  # NEW: on stocke la valeur

            mlflow.log_metric(f"{report_prefix}_drift_rate", drift_rate)
            mlflow.log_artifact(csv_report)
            mlflow.log_artifact(html_report)

            THRESHOLD = 0.3
            if drift_rate > THRESHOLD:
                backup_path = reference_path.replace(
                    ".csv",
                    f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                shutil.copy(reference_path, backup_path)
                current_df.to_csv(reference_path, index=False)

                print(f"🔁 Mise à jour référence ({report_prefix}) car drift > {THRESHOLD:.0%}")
                print(f"📦 Ancienne référence sauvegardée : {backup_path}")
            else:
                print(f"✅ Pas de drift majeur pour {report_prefix}")

    # NEW: sauvegarder un petit résumé lisible par retrain_if_drift.py
    summary_path = os.path.join(reports_path, "drift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(drift_rates, f, indent=2)

    print(f"\n📄 Résumé du drift enregistré dans {summary_path}")
    print("🎯 Monitoring terminé.")


if __name__ == "__main__":
    main()