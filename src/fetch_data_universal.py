import os
import time
import random
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from io import StringIO


# CONFIGURATION

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

LEAGUES = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "ITA-Serie A": "I1",
    "GER-Bundesliga": "D1",
    "FRA-Ligue 1": "F1",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

# ===============================
# OUTILS
# ===============================
def get_all_seasons():
    """Retourne toutes les saisons depuis 1993 jusqu’à la saison actuelle."""
    current_year = datetime.now().year % 100
    # Si on est en 2025 → current_year = 25 → de 93 à 25 = vide, donc on ajuste
    if current_year < 93:  # pour les années 2000+
        years = list(range(93, 100)) + list(range(0, current_year + 1))
    else:
        years = list(range(93, current_year + 1))
    return [f"{y:02d}{(y + 1) % 100:02d}" for y in years]


def safe_download(url):
    for attempt in range(1, 5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception as e:
            wait = 1.5 ** attempt + random.random()
            print(f"   Tentative {attempt}/4 échouée ({e}) — retry dans {wait:.1f}s")
            time.sleep(wait)
    return None


# LECTURE CSV

def fetch_league_data(league_name, code, season):
    """Télécharge un CSV pour une ligue et saison donnée."""
    url = f"{BASE_URL}/{season}/{code}.csv"
    print(f" {league_name} ({season}) — {url}")
    csv_text = safe_download(url)
    if not csv_text:
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        return pd.DataFrame()

    # Nettoyage minimal
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "homeTeam",
        "AwayTeam": "awayTeam",
        "FTHG": "homeScore",
        "FTAG": "awayScore",
    })
    df["league"] = league_name
    df["season"] = season
    return df[["date", "homeTeam", "awayTeam", "homeScore", "awayScore", "league", "season"]]


# MAIN

def main():
    print(" Début collecte Football-Data —", datetime.now().isoformat())

    all_matches = []
    all_stats = []

    seasons = get_all_seasons()
    print(f" Saisons ciblées: {len(seasons)} ({seasons[0]} → {seasons[-1]})")

    for league_name, code in LEAGUES.items():
        for season in seasons:
            df = fetch_league_data(league_name, code, season)
            if df.empty:
                continue
            all_matches.append(df)

            # Statistiques simplifiées
            tmp = df.dropna(subset=["homeScore", "awayScore"]).copy()
            tmp["homeScore"] = pd.to_numeric(tmp["homeScore"], errors="coerce")
            tmp["awayScore"] = pd.to_numeric(tmp["awayScore"], errors="coerce")

            home = tmp.groupby("homeTeam").agg(
                matches_home=("homeTeam", "count"),
                goals_for_home=("homeScore", "sum"),
                goals_against_home=("awayScore", "sum"),
            ).reset_index().rename(columns={"homeTeam": "team"})

            away = tmp.groupby("awayTeam").agg(
                matches_away=("awayTeam", "count"),
                goals_for_away=("awayScore", "sum"),
                goals_against_away=("homeScore", "sum"),
            ).reset_index().rename(columns={"awayTeam": "team"})

            merged = pd.merge(home, away, on="team", how="outer").fillna(0)
            merged["league"] = league_name
            merged["season"] = season
            merged["matches_played"] = merged["matches_home"] + merged["matches_away"]
            merged["goals_for"] = merged["goals_for_home"] + merged["goals_for_away"]
            merged["goals_against"] = merged["goals_against_home"] + merged["goals_against_away"]

            all_stats.append(merged)
            time.sleep(random.uniform(0.3, 0.9))

    # Sauvegarde
    if all_matches:
        full_matches = pd.concat(all_matches, ignore_index=True)
        full_matches.to_csv(RAW_PATH / "schedule_multi_leagues.csv", index=False)
        print(f" Matchs sauvegardés : {len(full_matches)} lignes")
    else:
        print(" Aucun match trouvé")

    if all_stats:
        full_stats = pd.concat(all_stats, ignore_index=True)
        full_stats.to_csv(RAW_PATH / "team_stats_multi_leagues.csv", index=False)
        print(f" Statistiques sauvegardées : {len(full_stats)} lignes")
    else:
        print(" Aucune statistique générée")

    print("\n Collecte terminée —", datetime.now().isoformat())


if __name__ == "__main__":
    main()
