import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import ks_2samp
import mlflow

try:
    from google.cloud import storage
except Exception:  # pragma: no cover
    storage = None


# CONFIG (env-friendly)


DRIFT_GCS_BUCKET = os.getenv("DRIFT_GCS_BUCKET", "reference_drift")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.30"))
RECENT_DAYS = int(os.getenv("DRIFT_RECENT_DAYS", "7"))

REPORTS_PATH = os.getenv("DRIFT_REPORTS_PATH", "reports")
PROCESSED_PATH = os.getenv("DRIFT_PROCESSED_PATH", "data/processed")
RAW_PATH = os.getenv("DRIFT_RAW_PATH", "data/raw")

os.makedirs(REPORTS_PATH, exist_ok=True)


# LOGGING


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# GCS HELPERS FOR REFERENCES


def get_gcs_client():
    """Return a GCS client or None if not available/misconfigured."""
    if not DRIFT_GCS_BUCKET or storage is None:
        return None
    try:
        return storage.Client()
    except Exception as e:  # pragma: no cover
        log(f"⚠️ Impossible de créer un client GCS : {e}")
        return None


def _gcs_blob_name(local_path: str) -> str:
    """
    Build the blob name in GCS from the local path.
    We keep the relative path at the root of the bucket.
    """
    rel_path = os.path.relpath(local_path).replace("\\", "/")
    return rel_path


def download_reference_from_gcs(local_path: str):
    """If a reference file exists in GCS, download it to local_path."""
    client = get_gcs_client()
    if client is None:
        return

    bucket = client.bucket(DRIFT_GCS_BUCKET)
    blob_name = _gcs_blob_name(local_path)
    blob = bucket.blob(blob_name)

    try:
        exists = blob.exists()
    except Exception as e:  # pragma: no cover
        log(f" Erreur vérification existence GCS : {e}")
        return

    if not exists:
        log(f" Aucune référence distante pour gs://{DRIFT_GCS_BUCKET}/{blob_name}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    log(f" Référence téléchargée depuis gs://{DRIFT_GCS_BUCKET}/{blob_name}")


def upload_reference_to_gcs(local_path: str):
    """Upload the local reference file to GCS (overwriting existing)."""
    if not os.path.exists(local_path):
        log(f" Impossible d’uploader, fichier introuvable : {local_path}")
        return

    client = get_gcs_client()
    if client is None:
        return

    bucket = client.bucket(DRIFT_GCS_BUCKET)
    blob_name = _gcs_blob_name(local_path)
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_filename(local_path)
        log(f" Référence uploadée vers gs://{DRIFT_GCS_BUCKET}/{blob_name}")
    except Exception as e:  # pragma: no cover
        log(f"⚠️ Échec upload GCS : {e}")



# FEATURE SELECTION (fallback)


def get_numeric_features(df: pd.DataFrame, dataset_name: str) -> List[str]:
    """Return a cleaned list of numeric features for drift detection."""

    numeric_cols = [c for c in df.columns if df[c].dtype != "object"]

    exclude_cols = [
        "player", "player_name", "team", "team_name",
        "id", "match_id", "player_id",
        "date" 
    ]

    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not numeric_cols:
        log(f" Aucun feature numérique utile trouvé pour {dataset_name}")
        return []

    log(f" {dataset_name} : {len(numeric_cols)} features auto → {numeric_cols}")
    return numeric_cols


# KS DRIFT CORE


def run_ks_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: List[str],
    min_samples: int = 20,
) -> Tuple[float, pd.DataFrame]:
    """Run KS drift on a list of features and return drift_rate + report df."""
    results = []

    for col in features:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref = ref_df[col].dropna()
        cur = cur_df[col].dropna()

        if len(ref) < min_samples or len(cur) < min_samples:
            continue

        stat, p_value = ks_2samp(ref, cur)
        drift = p_value < 0.05

        results.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(drift),
        })

    report_df = pd.DataFrame(results)

    if report_df.empty:
        return 0.0, report_df

    drift_rate = float(report_df["drift_detected"].mean())
    return drift_rate, report_df



# REPORTING


def save_reports(name: str, report_df: pd.DataFrame) -> Tuple[str, str]:
    """Save CSV + HTML drift report and return their paths."""
    csv_path = os.path.join(REPORTS_PATH, f"{name}_drift_report.csv")
    html_path = os.path.join(REPORTS_PATH, f"{name}_drift_report.html")

    report_df.to_csv(csv_path, index=False)

    drift_count = int(report_df["drift_detected"].sum()) if not report_df.empty else 0
    total = int(len(report_df))
    drift_rate = (drift_count / total) if total else 0.0

    html_content = f"""
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <title>Data Drift Report - {name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1> Data Drift Report — {name}</h1>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Total tested features:</b> {total}</p>
        <p><b>Drift detected:</b> {drift_count} ({drift_rate:.1%})</p>
        <hr>
        {report_df.to_html(index=False)}
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return csv_path, html_path



# WINDOW EXTRACTORS


def window_full(df: pd.DataFrame, _: Dict) -> pd.DataFrame:
    return df


def window_recent_by_date(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Filter rows where date >= max(date) - RECENT_DAYS."""
    date_col = meta.get("date_col", "date")
    if date_col not in df.columns:
        return df

    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    max_date = df[date_col].max()

    if pd.isna(max_date):
        return df

    cut = max_date - timedelta(days=RECENT_DAYS)
    return df[df[date_col] >= cut]


def window_tail(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    n = int(meta.get("tail_n", 500))
    return df.tail(n)



# DATASET CONFIG


DatasetConfig = Dict[str, object]

DATASETS: Dict[str, DatasetConfig] = {
    # Matches (recent by date)
    "model1_clean": {
        "current_path": os.path.join(PROCESSED_PATH, "clean_matches.csv"),
        "reference_path": os.path.join(PROCESSED_PATH, "clean_matches_reference.csv"),
        "features": [
            "home_goals", "away_goals",
            "home_goals_for", "away_goals_for",
            "home_goals_against", "away_goals_against",
            "home_goals_diff", "away_goals_diff",
        ],
        "window_fn": window_recent_by_date,
        "meta": {"date_col": "date"},
    },

    # Player strengths (recent tail)
    "player_strengths": {
        "current_path": os.path.join(PROCESSED_PATH, "player_strengths.csv"),
        "reference_path": os.path.join(PROCESSED_PATH, "player_strengths_reference.csv"),
        "features": [
            "goals", "assists", "xG", "xAG",
            "minutes", "goals_per90", "assists_per90",
            "xG_per90", "player_score",
        ],
        "window_fn": window_tail,
        "meta": {"tail_n": 500},
    },

    # Team season stats (full)
    "team_season_stats": {
        "current_path": os.path.join(RAW_PATH, "team_season_stats_model2.csv"),
        "reference_path": os.path.join(RAW_PATH, "team_season_stats_reference.csv"),
        "features": ["players_used", "Age", "Poss"],
        "window_fn": window_full,
        "meta": {},
    },

    # Optional extra dataset from the old script (auto numeric fallback)
    "team_match_stats": {
        "current_path": os.path.join(RAW_PATH, "team_match_stats_model2.csv"),
        "reference_path": os.path.join(RAW_PATH, "team_match_stats_reference.csv"),
        "features": None,  # auto numeric
        "window_fn": window_full,
        "meta": {},
    },
}


# SINGLE DATASET MONITOR


def monitor_dataset(name: str, cfg: DatasetConfig) -> Optional[float]:
    current_path = str(cfg["current_path"])
    reference_path = str(cfg["reference_path"])
    window_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame] = cfg.get("window_fn", window_full)  # type: ignore
    meta: Dict = cfg.get("meta", {})  # type: ignore

    if not os.path.exists(current_path):
        log(f" Fichier manquant : {current_path}")
        return None

    log(f"\n Vérification du drift pour : {name}")

    # Load current
    # Parse dates only when likely needed
    try:
        if meta.get("date_col"):
            current_df = pd.read_csv(current_path, parse_dates=[meta["date_col"]])
        else:
            current_df = pd.read_csv(current_path)
    except Exception:
        current_df = pd.read_csv(current_path)

    # Apply window
    current_window = window_fn(current_df, meta)

    # Determine features
    features_cfg = cfg.get("features")
    if features_cfg is None:
        # Try to coerce to numeric where possible to help auto scan
        tmp = current_window.copy()
        for col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="ignore")
        features = get_numeric_features(tmp, name)
        current_window = tmp
    else:
        features = list(features_cfg)  # type: ignore

    if not features:
        log(f" Aucune feature à tester pour {name}")
        return None

    # Try to fetch reference from GCS first
    download_reference_from_gcs(reference_path)

    # First run -> create reference from current window
    if not os.path.exists(reference_path):
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        current_window[features].to_csv(reference_path, index=False)
        log(f" Référence créée : {reference_path}")
        upload_reference_to_gcs(reference_path)
        return None

    # Load reference
    reference_df = pd.read_csv(reference_path)

    # Ensure reference contains only relevant cols if curated
    # (Avoids schema noise over time)
    missing = [f for f in features if f not in reference_df.columns]
    if missing:
        log(f" Référence {name} ne contient pas toutes les features : {missing}")

    # Align frames on features that exist in both
    common_features = [f for f in features if f in reference_df.columns and f in current_window.columns]

    if not common_features:
        log(f" Aucune feature commune pour {name}")
        return None

    # Run KS
    drift_rate, report_df = run_ks_drift(reference_df, current_window, common_features)

    # Save reports
    csv_path, html_path = save_reports(name, report_df)

    # Log MLflow
    mlflow.log_metric(f"{name}_drift_rate", drift_rate)
    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(html_path)

    log(f" {name} drift = {drift_rate:.1%} ({int(report_df.get('drift_detected', pd.Series(dtype=bool)).sum())}/{len(report_df)} features)")

    # Update reference if above threshold
    if drift_rate > DRIFT_THRESHOLD:
        backup_path = reference_path.replace(
            ".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        try:
            shutil.copy(reference_path, backup_path)
        except Exception:
            pass

        current_window[common_features].to_csv(reference_path, index=False)

        log(
            f" Mise à jour référence ({name}) "
            f"car drift {drift_rate:.1%} > seuil {DRIFT_THRESHOLD:.0%}"
        )
        log(f" Ancienne référence sauvegardée : {backup_path}")

        upload_reference_to_gcs(reference_path)
    else:
        log(f" Pas de drift majeur pour {name} (≤ {DRIFT_THRESHOLD:.0%})")

    return drift_rate



# MAIN


def main():
    log(f"DRIFT MONITORING — seuil = {DRIFT_THRESHOLD:.0%} | fenêtre récente = {RECENT_DAYS} jours")
    log(f"GCS bucket références = {DRIFT_GCS_BUCKET}")

    mlflow.set_experiment("football_prediction_mlops")
    drift_rates: Dict[str, float] = {}

    with mlflow.start_run(run_name="weekly_data_drift_with_gcs"):
        for name, cfg in DATASETS.items():
            try:
                rate = monitor_dataset(name, cfg)
                if rate is not None:
                    drift_rates[name] = rate
            except Exception as e:
                log(f" Erreur monitoring {name} : {e}")

        # Save and log summary JSON for retraining pipelines
        summary_path = os.path.join(REPORTS_PATH, "drift_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(drift_rates, f, indent=2)

        mlflow.log_artifact(summary_path)
        log(f" Résumé du drift enregistré dans {summary_path}")

    log(" Monitoring terminé.")


if __name__ == "__main__":
    main()
