from __future__ import annotations

import os
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st

BASE_PATH = Path(__file__).resolve().parent
DATA_DIR = BASE_PATH / "data"
REPORTS_DIR = BASE_PATH / "reports"
APP_DIR = BASE_PATH / "app"
DVC_FILE = BASE_PATH / "dvc.yaml"
DEFAULT_API_URL = os.getenv("FOOTBALL_API_URL", "http://localhost:8000")

METRIC_FILES = {
    "model1": REPORTS_DIR / "model1_metrics.json",
    "model2": REPORTS_DIR / "model2_metrics.json",
    "model3_player_mode": REPORTS_DIR / "model3_player_mode_metrics.json",
}

PREDICTION_FILES = {
    "model1": DATA_DIR / "predictions" / "predicted_matches.csv",
    "model2": DATA_DIR / "predictions" / "model2_predictions.csv",
    "model3_player_mode": DATA_DIR / "predictions" / "model3_players_output.csv",
}

STAGE_GROUPS = {
    "Model 1": {"fetch_data", "preprocess", "train", "predict"},
    "Model 2": {
        "extract_matches_model2",
        "extract_player_stats_model2",
        "extract_team_stats_model2",
        "build_team_name_mapping",
        "preprocess_model2",
        "train_model2",
        "predict_model2",
    },
    "Model 3": {"build_player_strengths", "predict_model3_players", "eval_model3_player_mode"},
    "Data Drift": {"data_drift"},
    "Governance": {"select_champion"},
}


def stage_group(name: str) -> str:
    for label, names in STAGE_GROUPS.items():
        if name in names:
            return label
    if name.startswith("extract"):
        return "Extraction"
    return "Pipeline"


@st.cache_data(show_spinner=False)
def load_dataframe(
    path: Path,
    *,
    nrows: int | None = None,
    parse_dates: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=nrows, parse_dates=list(parse_dates or []))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json_file(path: Path) -> Dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics_table() -> pd.DataFrame:
    rows = []
    for name, path in METRIC_FILES.items():
        data = load_json_file(path)
        if data:
            rows.append({"model": name} | data)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_drift_summary() -> pd.DataFrame:
    path = REPORTS_DIR / "drift_summary.json"
    data = load_json_file(path) or {}
    if not data:
        return pd.DataFrame(columns=["dataset", "drift_rate"])
    return pd.DataFrame(data.items(), columns=["dataset", "drift_rate"]).sort_values(
        "drift_rate", ascending=False
    )


@st.cache_data(show_spinner=False)
def load_drift_reports() -> Dict[str, pd.DataFrame]:
    reports: Dict[str, pd.DataFrame] = {}
    if not REPORTS_DIR.exists():
        return reports
    for csv_path in REPORTS_DIR.glob("*_drift_report.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        dataset = csv_path.stem.replace("_drift_report", "")
        reports[dataset] = df
    return reports


@st.cache_data(show_spinner=False)
def load_dvc_stages() -> pd.DataFrame:
    if not DVC_FILE.exists():
        return pd.DataFrame()
    try:
        import yaml
    except Exception:
        return pd.DataFrame()
    try:
        with DVC_FILE.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return pd.DataFrame()

    stage_rows = []
    for name, cfg in (doc.get("stages") or {}).items():
        deps = cfg.get("deps") or []
        outs = cfg.get("outs") or []
        stage_rows.append(
            {
                "stage": name,
                "group": stage_group(name),
                "cmd": cfg.get("cmd", ""),
                "deps": "\n".join(str(d) for d in deps),
                "outs": "\n".join(str(o) for o in outs),
                "frozen": bool(cfg.get("frozen", False)),
            }
        )
    return pd.DataFrame(stage_rows)


def collect_team_options(*dfs: pd.DataFrame | None) -> list[str]:
    teams = set()
    for df in dfs:
        if df is None:
            continue
        for col in ("home_team", "away_team", "team"):
            if col in df.columns:
                teams.update(df[col].dropna().astype(str))
    return sorted(t for t in teams if t.strip())


def build_roster_map(player_df: pd.DataFrame | None) -> Dict[str, list[str]]:
    roster: Dict[str, list[str]] = {}
    if player_df is None or not {"team", "player"}.issubset(player_df.columns):
        return roster
    grouped = player_df.groupby("team")["player"]
    for team, series in grouped:
        players = sorted({str(p).strip() for p in series.dropna() if str(p).strip()})
        if team and players:
            roster[str(team)] = players
    return roster


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}".replace(",", " ")


def file_timestamp(path: Path) -> str:
    if not path.exists():
        return "-"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")


def api_post(api_url: str, path: str, payload: Dict) -> Dict | None:
    if not api_url:
        st.error("Aucune URL API configuree.")
        return None
    url = api_url.rstrip("/") + path
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Appel {url} en erreur : {exc}")
        return None


def filter_predictions(
    df: pd.DataFrame | None,
    selected_teams: Tuple[str, ...],
    date_range: Tuple[datetime, datetime] | None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if selected_teams:
        mask = data["home_team"].isin(selected_teams) | data["away_team"].isin(selected_teams)
        data = data[mask]

    if date_range and "date" in data.columns:
        start, end = date_range
        data = data[(data["date"] >= start) & (data["date"] <= end)]

    return data


def sidebar_filters(
    clean_df: pd.DataFrame | None,
    pred_df: pd.DataFrame | None,
    team_choices: Iterable[str] | None = None,
):
    st.sidebar.header("Filtres globaux")

    if team_choices is not None:
        team_options = sorted(set(team_choices))
    else:
        team_values = []
        for df in (clean_df, pred_df):
            if df is None:
                continue
            for col in ("home_team", "away_team"):
                if col in df.columns:
                    team_values.extend(df[col].dropna().astype(str).tolist())
        team_options = sorted(set(team_values))
    selected_teams = st.sidebar.multiselect(
        "Filtrer sur une ou plusieurs equipes",
        team_options,
        default=[],
        placeholder="Toutes les equipes",
    )

    date_range = None
    if pred_df is not None and "date" in pred_df.columns:
        dates = pd.to_datetime(pred_df["date"], errors="coerce").dropna()
        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()
            date_selection = st.sidebar.date_input(
                "Plage de dates (predictions model1)",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_selection, tuple) and len(date_selection) == 2:
                start_date = datetime.combine(date_selection[0], datetime.min.time())
                end_date = datetime.combine(date_selection[1], datetime.max.time())
                date_range = (start_date, end_date)

    st.sidebar.caption(
        "Les filtres s'appliquent aux tableaux et graphiques des predictions dans les onglets."
    )
    st.sidebar.header("API temps reel")
    api_url = st.sidebar.text_input("URL FastAPI", value=DEFAULT_API_URL)
    st.sidebar.caption("Demarrer l'API : `uvicorn app.main:app --host 0.0.0.0 --port 8000`")
    return tuple(selected_teams), date_range, api_url.strip()


def render_overview_tab(
    champion_cfg: Dict | None,
    clean_df: pd.DataFrame | None,
    pred_df: pd.DataFrame | None,
    drift_summary_df: pd.DataFrame,
    dvc_df: pd.DataFrame,
):
    st.subheader("Synthese operationnelle")

    col1, col2, col3 = st.columns(3)
    col1.metric("Matchs historicises", format_int(len(clean_df) if clean_df is not None else None))
    col2.metric("Predictions stockees", format_int(len(pred_df) if pred_df is not None else None))
    high_drift = (
        (
            drift_summary_df[drift_summary_df["drift_rate"] > 0.30]["dataset"].count()
            if not drift_summary_df.empty
            else 0
        )
    )
    col3.metric("Datasets au-dessus du seuil", format_int(high_drift))

    st.markdown("---")
    st.subheader("Champion actif et gouvernance")
    if champion_cfg:
        meta_cols = st.columns(3)
        meta_cols[0].metric("Champion", champion_cfg.get("champion_key", "-"))
        if champion_cfg.get("metrics"):
            metric_priority = champion_cfg.get("metric_priority", [])
            if metric_priority:
                meta_cols[1].metric("Priorite", " > ".join(metric_priority))
        generated_at = champion_cfg.get("generated_at")
        meta_cols[2].metric("Dernier arbitrage", generated_at or "-")
        st.json(champion_cfg.get("metrics", {}))
    else:
        st.info("Aucune configuration de champion trouvee. Lance `python src/select_champion.py`.")

    st.markdown("---")
    st.subheader("Pipeline DVC")
    if dvc_df.empty:
        st.warning("Fichier `dvc.yaml` non disponible ou illisible.")
    else:
        chart_data = (
            dvc_df.groupby("group")
            .size()
            .reset_index(name="stages")
            .sort_values("stages", ascending=False)
        )
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(x="group", y="stages", tooltip=["group", "stages"])
        )
        st.altair_chart(chart, use_container_width=True)
        stage_table = dvc_df.assign(
            frozen_label=dvc_df["frozen"].map({True: "Oui", False: "Non"})
        )[["stage", "group", "cmd", "deps", "outs", "frozen_label"]]
        st.dataframe(stage_table, use_container_width=True)


def render_models_tab(metrics_df: pd.DataFrame, champion_cfg: Dict | None):
    st.subheader("Performances des modeles")
    if metrics_df.empty:
        st.warning("Aucune metrique dans `reports/`. Lance les etapes `train*`.")
        return

    st.dataframe(metrics_df, use_container_width=True)
    metric_cols = [c for c in metrics_df.columns if c not in {"model"}]
    if not metric_cols:
        st.info("Aucune colonne de metrique disponible.")
        return
    selected_metric = st.selectbox("Metrique a comparer", metric_cols, index=0)
    chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x="model",
            y=alt.Y(selected_metric, title=selected_metric),
            color=alt.condition(
                alt.datum.model == (champion_cfg or {}).get("champion_key", ""),
                alt.value("#0f62fe"),
                alt.value("#888"),
            ),
            tooltip=list(metrics_df.columns),
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Details")
    for _, row in metrics_df.iterrows():
        is_champion = row["model"] == (champion_cfg or {}).get("champion_key", "")
        with st.expander(f"{row['model']} {'(champion)' if is_champion else ''}"):
            metrics = row.drop(labels=["model"]).to_dict()
            st.json(metrics)


def render_predictions_tab(
    pred_df: pd.DataFrame | None,
    model2_df: pd.DataFrame | None,
    model3_df: pd.DataFrame | None,
    api_url: str,
    team_options: list[str],
    roster_map: Dict[str, list[str]],
) -> None:
    st.subheader("Predictions Model 1 (regression)")
    if pred_df is None or pred_df.empty:
        st.info("Pas de predictions `data/predictions/predicted_matches.csv`.")
    else:
        agg_cols = st.columns(3)
        agg_cols[0].metric("Matches visibles", format_int(len(pred_df)))
        if {"pred_home_goals", "pred_away_goals"}.issubset(pred_df.columns):
            agg_cols[1].metric(
                "Buts domicile moyens",
                f"{pred_df['pred_home_goals'].mean():.2f}",
            )
            agg_cols[2].metric(
                "Buts exterieur moyens",
                f"{pred_df['pred_away_goals'].mean():.2f}",
            )
        st.dataframe(pred_df.head(200), use_container_width=True)
        if "predicted_result" in pred_df.columns:
            dist = (
                pred_df["predicted_result"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "resultat", "predicted_result": "matches"})
            )
            if dist.empty:
                st.info("Aucun resultat a visualiser avec les filtres courants.")
            else:
                chart = (
                    alt.Chart(dist)
                    .mark_bar()
                    .encode(
                        x=alt.X("resultat:N", title="Resultat"),
                        y=alt.Y("matches:Q", title="Nombre de matchs"),
                        tooltip=["resultat", "matches"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Predictions Model 2 (classification)")
    if model2_df is None or model2_df.empty:
        st.info("Pas de `data/predictions/model2_predictions.csv`.")
    else:
        st.dataframe(model2_df.head(200), use_container_width=True)
        required_cols = {"proba_away_win", "proba_draw", "proba_home_win"}
        if required_cols.issubset(model2_df.columns):
            prob_chart = (
                alt.Chart(model2_df)
                .transform_fold(
                    sorted(required_cols),
                    as_=["scenario", "value"],
                )
                .mark_bar()
                .encode(x="scenario", y="average(value):Q", color="scenario")
            )
            st.altair_chart(prob_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Predictions Model 3 (mode joueurs)")
    if model3_df is None or model3_df.empty:
        st.info("Pas de `data/predictions/model3_players_output.csv`.")
    else:
        st.dataframe(model3_df.head(200), use_container_width=True)
        if {"home_strength", "away_strength"}.issubset(model3_df.columns):
            scatter = (
                alt.Chart(model3_df)
                .mark_circle()
                .encode(
                    x="home_strength",
                    y="away_strength",
                    color="prediction",
                    tooltip=["home_team", "away_team", "prediction"],
                )
            )
            st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Predictions live via API")
    render_live_api_section(api_url, team_options, roster_map)


def render_live_api_section(api_url: str, team_options: list[str], roster_map: Dict[str, list[str]]) -> None:
    if not api_url:
        st.info("Renseigne l'URL FastAPI dans la barre laterale pour activer les appels live.")
        return

    if not team_options:
        st.info("Aucune equipe n'a pu etre extraite des donnees pour alimenter les formulaires.")
        return

    st.caption(
        "Lance l'API (`uvicorn app.main:app --host 0.0.0.0 --port 8000`) puis utilise les formulaires ci-dessous."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Model 1 (regression)")
        with st.form("api_model1_form"):
            home_team = st.selectbox(
                "Equipe domicile",
                options=team_options,
                key="api_model1_home",
            )
            away_team = st.selectbox(
                "Equipe exterieur",
                options=team_options,
                key="api_model1_away",
            )
            submitted = st.form_submit_button("Predire via /predict/model1")
            if submitted:
                if home_team == away_team:
                    st.warning("Choisis deux equipes differentes.")
                else:
                    with st.spinner("Appel API..."):
                        result = api_post(
                            api_url,
                            "/predict/model1",
                            {"home_team": home_team, "away_team": away_team},
                        )
                    if result:
                        st.json(result)

    with col_b:
        st.markdown("#### Model 2 (classification)")
        with st.form("api_model2_form"):
            home_team = st.selectbox(
                "Equipe domicile",
                options=team_options,
                key="api_model2_home",
            )
            away_team = st.selectbox(
                "Equipe exterieur",
                options=team_options,
                key="api_model2_away",
            )
            submitted = st.form_submit_button("Predire via /predict/model2")
            if submitted:
                if home_team == away_team:
                    st.warning("Choisis deux equipes differentes.")
                else:
                    with st.spinner("Appel API..."):
                        result = api_post(
                            api_url,
                            "/predict/model2",
                            {"home_team": home_team, "away_team": away_team},
                        )
                    if result:
                        st.json(result)

    st.markdown("#### Model 3 (mode joueurs)")
    with st.form("api_model3_form"):
        home_team = st.selectbox(
            "Equipe domicile",
            options=team_options,
            key="api_model3_home",
        )
        away_team = st.selectbox(
            "Equipe exterieur",
            options=team_options,
            key="api_model3_away",
        )
        home_player_options = roster_map.get(home_team, [])
        away_player_options = roster_map.get(away_team, [])
        home_players = st.multiselect(
            "Selectionne 11 joueurs domicile",
            options=home_player_options,
            key="api_model3_home_players",
        )
        away_players = st.multiselect(
            "Selectionne 11 joueurs exterieur",
            options=away_player_options,
            key="api_model3_away_players",
        )
        submitted = st.form_submit_button("Predire via /predict/model3")
        if submitted:
            if home_team == away_team:
                st.warning("Choisis deux equipes differentes.")
            elif len(home_players) != 11 or len(away_players) != 11:
                st.warning("Selectionne exactement 11 joueurs pour chaque equipe.")
            else:
                payload = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_players": list(home_players),
                    "away_players": list(away_players),
                }
                with st.spinner("Appel API..."):
                    result = api_post(api_url, "/predict/model3", payload)
                if result:
                    st.json(result)


def render_drift_tab(
    drift_summary_df: pd.DataFrame,
    drift_reports: Dict[str, pd.DataFrame],
):
    st.subheader("Surveillance du drift")
    if drift_summary_df.empty:
        st.info(
            "Aucun resume dans `reports/drift_summary.json`. Lance `python src/monitor_drift.py`."
        )
    else:
        threshold = st.slider("Seuil visuel", 0.0, 1.0, 0.30, 0.01)
        chart = (
            alt.Chart(drift_summary_df)
            .mark_bar()
            .encode(
                x="dataset",
                y=alt.Y("drift_rate", axis=alt.Axis(format="%")),
                color=alt.condition(
                    alt.datum.drift_rate > threshold,
                    alt.value("#fa4d56"),
                    alt.value("#24a148"),
                ),
                tooltip=["dataset", alt.Tooltip("drift_rate", format=".1%")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(drift_summary_df, use_container_width=True)

    st.markdown("### Rapports detailles")
    if not drift_reports:
        st.info("Aucun fichier `*_drift_report.csv` trouve.")
        return
    dataset = st.selectbox("Selectionner un dataset", sorted(drift_reports.keys()))
    report = drift_reports.get(dataset)
    if report is None or report.empty:
        st.info("Rapport vide.")
        return
    st.dataframe(report, use_container_width=True)
    if {"feature", "ks_statistic"}.issubset(report.columns):
        feature_chart = (
            alt.Chart(report)
            .mark_bar()
            .encode(
                x="feature",
                y="ks_statistic",
                color=alt.condition(
                    alt.datum.drift_detected == True,  # noqa: E712
                    alt.value("#d62728"),
                    alt.value("#1f77b4"),
                ),
            )
        )
        st.altair_chart(feature_chart, use_container_width=True)


def render_data_tab(
    clean_df: pd.DataFrame | None,
    player_df: pd.DataFrame | None,
    training_df: pd.DataFrame | None,
):
    st.subheader("Jeux de donnees a disposition")
    with st.expander("data/processed/clean_matches.csv"):
        if clean_df is None:
            st.info("Fichier non disponible.")
        else:
            st.dataframe(clean_df.head(200), use_container_width=True)

    with st.expander("data/processed/player_strengths.csv"):
        if player_df is None:
            st.info("Fichier non disponible.")
        else:
            st.dataframe(player_df.head(200), use_container_width=True)

    with st.expander("data/processed/model2_training_dataset.csv"):
        if training_df is None:
            st.info("Fichier non disponible.")
        else:
            st.dataframe(training_df.head(200), use_container_width=True)

    st.markdown("### Rappels operatoires")
    st.code("dvc repro", language="bash")
    st.code("mlflow ui --backend-store-uri ./mlruns", language="bash")
    st.code("uvicorn app.main:app --host 0.0.0.0 --port 8000", language="bash")


def main():
    alt.data_transformers.disable_max_rows()
    st.set_page_config(page_title="Football Prediction Control Center", layout="wide")
    st.title("Control Center Football Prediction")
    st.caption(
        "Dashboard unifie couvrant pipeline DVC, metriques modeles, predictions, drift et jeux de donnees."
    )

    clean_df = load_dataframe(DATA_DIR / "processed" / "clean_matches.csv", parse_dates=["date"])
    pred_df = load_dataframe(PREDICTION_FILES["model1"])
    model2_pred_df = load_dataframe(PREDICTION_FILES["model2"])
    model3_pred_df = load_dataframe(PREDICTION_FILES["model3_player_mode"])
    player_df = load_dataframe(DATA_DIR / "processed" / "player_strengths.csv")
    training_df = load_dataframe(
        DATA_DIR / "processed" / "model2_training_dataset.csv", nrows=500
    )

    team_options = collect_team_options(clean_df, pred_df, player_df)
    roster_map = build_roster_map(player_df)

    metrics_df = load_metrics_table()
    drift_summary_df = load_drift_summary()
    drift_reports = load_drift_reports()
    dvc_df = load_dvc_stages()
    champion_cfg = load_json_file(APP_DIR / "champion_config.json")

    selected_teams, date_range, api_url = sidebar_filters(
        clean_df, pred_df, team_choices=team_options
    )
    filtered_pred_df = filter_predictions(pred_df, selected_teams, date_range)

    tabs = st.tabs(
        [
            "Synthese",
            "Modeles & metriques",
            "Predictions",
            "Data drift",
            "Datasets & runbook",
        ]
    )

    with tabs[0]:
        render_overview_tab(champion_cfg, clean_df, filtered_pred_df, drift_summary_df, dvc_df)
    with tabs[1]:
        render_models_tab(metrics_df, champion_cfg)
    with tabs[2]:
        render_predictions_tab(
            filtered_pred_df, model2_pred_df, model3_pred_df, api_url, team_options, roster_map
        )
    with tabs[3]:
        render_drift_tab(drift_summary_df, drift_reports)
    with tabs[4]:
        render_data_tab(clean_df, player_df, training_df)


if __name__ == "__main__":
    main()
