import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def load_raw_data():
    log("Loading raw data for Model 2...")

    schedule = pd.read_csv("data/raw/schedule_model2.csv")
    players = pd.read_csv("data/raw/player_season_stats_model2.csv")
    mapping = pd.read_csv("data/team_name_mapping.csv")

    log(f"Schedule: {schedule.shape}")
    log(f"Player stats: {players.shape}")
    log(f"Mapping: {mapping.shape}")

    return schedule, players, mapping


# CLEAN SCHEDULE

def clean_score(score_str):
    if pd.isna(score_str):
        return None, None

    s = str(score_str)

    # remove parentheses like "(5)" "(a.e.t.)"
    while "(" in s and ")" in s:
        start = s.index("(")
        end = s.index(")") + 1
        s = s.replace(s[start:end], "").strip()

    if "–" not in s:
        return None, None

    try:
        a, b = s.split("–")
        return int(a.strip()), int(b.strip())
    except:
        return None, None


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    log("Preparing schedule...")

    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    schedule["home_score"], schedule["away_score"] = zip(
        *schedule["score"].apply(clean_score)
    )

    schedule["result"] = schedule.apply(
        lambda r: np.sign(r["home_score"] - r["away_score"])
        if pd.notna(r["home_score"])
        else np.nan,
        axis=1,
    )

    log("Schedule prepared.")
    return schedule


# PLAYER SCORING

def compute_player_scores(players: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    log("Computing player scores...")

    df = players.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    # convert all numerics except identifiers
    numeric_cols = [c for c in df.columns if c not in ["league","season","team","player","nation","pos","age","born"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["team"] = df["team"].astype(str).str.strip()

    # map team names
    mapping_dict = dict(zip(mapping["schedule_name"], mapping["stats_name"]))
    df["team_clean"] = df["team"].map(mapping_dict).fillna(df["team"])

    # fbref columns we use
    col_goals = "Performance"
    col_assists = "Performance.1"
    col_xg = "Expected"
    col_sot = "Performance.3"
    col_shots = "Performance.2"
    col_tackles = "Performance.4"
    col_inter = "Performance.5"
    col_prog = "Progression"
    col_minutes = "Playing Time"

    df["player_score"] = 0.0

    # GK
    mask_gk = df["pos"].str.contains("GK", na=False)
    df.loc[mask_gk, "player_score"] = (
        df[col_minutes] * 0.01
    )

    # DF
    mask_df = df["pos"].str.contains("DF", na=False)
    df.loc[mask_df, "player_score"] = (
        df[col_tackles] * 2 +
        df[col_inter] * 2 +
        df[col_assists] * 3 +
        df[col_goals] * 5
    )

    # MF
    mask_mf = df["pos"].str.contains("MF", na=False)
    df.loc[mask_mf, "player_score"] = (
        df[col_prog] * 2 +
        df[col_assists] * 3 +
        df[col_goals] * 4 +
        df[col_inter] * 2
    )

    # FW
    mask_fw = df["pos"].str.contains("FW", na=False)
    df.loc[mask_fw, "player_score"] = (
        df[col_goals] * 6 +
        df[col_assists] * 3 +
        df[col_sot] * 2 +
        df[col_xg] * 2
    )

    # normalize
    scaler = MinMaxScaler()
    df["player_score"] = scaler.fit_transform(df[["player_score"]])

    log("Player scoring OK.")
    return df[["team_clean", "player", "pos", "player_score"]]


# TEAM STRENGTH

def aggregate_team_strength(player_scores: pd.DataFrame) -> pd.DataFrame:
    log("Aggregating team strengths...")

    team_strength = (
        player_scores.groupby("team_clean")["player_score"]
        .mean()
        .reset_index()
        .rename(columns={"team_clean": "team", "player_score": "team_strength"})
    )

    log(team_strength.head())
    return team_strength


# APPLY MAPPING

def apply_mapping(schedule, mapping):
    log("Applying team name mapping...")

    map_dict = dict(zip(mapping["schedule_name"], mapping["stats_name"]))

    schedule["home_team_clean"] = schedule["home_team"].map(map_dict).fillna(schedule["home_team"])
    schedule["away_team_clean"] = schedule["away_team"].map(map_dict).fillna(schedule["away_team"])

    return schedule


# MERGE FINAL DATASET

def build_final_dataset(schedule, team_strength):
    log("Merging schedule with team strengths...")

    df = schedule.copy()

    # MERGE HOME
    df = df.merge(
        team_strength.rename(columns={"team": "home_team_clean", "team_strength": "home_strength"}),
        on="home_team_clean",
        how="left"
    )

    # MERGE AWAY
    df = df.merge(
        team_strength.rename(columns={"team": "away_team_clean", "team_strength": "away_strength"}),
        on="away_team_clean",
        how="left"
    )

    df["strength_diff"] = df["home_strength"] - df["away_strength"]

    return df


def main():
    schedule, players, mapping = load_raw_data()

    schedule = prepare_schedule(schedule)
    schedule = apply_mapping(schedule, mapping)

    player_scores = compute_player_scores(players, mapping)
    team_strength = aggregate_team_strength(player_scores)

    dataset = build_final_dataset(schedule, team_strength)

    os.makedirs("data/processed", exist_ok=True)
    dataset.to_csv("data/processed/model2_preprocessed.csv", index=False)

    log("Saved → data/processed/model2_preprocessed.csv")


if __name__ == "__main__":
    main()
