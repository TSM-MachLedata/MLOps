import os
import pandas as pd
from datetime import datetime


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def extract_team_from_url(url):
    if pd.isna(url):
        return None
    last = url.split("/")[-1]          # ex: "Arsenal-Stats"
    team = last.replace("-Stats", "")  # "Arsenal"
    team = team.replace("-", " ")      # "Brighton-and-Hove-Albion" -> "Brighton and Hove Albion"
    return team.strip()


def main():
    schedule_path = "data/raw/schedule_model2.csv"
    team_stats_path = "data/raw/team_season_stats_model2.csv"
    out_path = "data/team_name_mapping.csv"

    if not os.path.exists(schedule_path):
        raise FileNotFoundError(schedule_path)
    if not os.path.exists(team_stats_path):
        raise FileNotFoundError(team_stats_path)

    log("Loading data...")
    schedule = pd.read_csv(schedule_path)
    team_stats = pd.read_csv(team_stats_path)

    # Extraire team depuis l'URL FBRef
    team_stats["team"] = team_stats["url"].apply(extract_team_from_url)

    # Toutes les équipes du calendrier
    schedule_teams = set(schedule["home_team"].dropna().unique()) | \
                     set(schedule["away_team"].dropna().unique())

    # Toutes les équipes trouvées dans les stats FBRef
    stats_teams = set(team_stats["team"].dropna().unique())

    log(f"Unique schedule teams: {len(schedule_teams)}")
    log(f"Unique stats teams: {len(stats_teams)}")

    rows = []

    # 1) mapping direct
    for name in sorted(schedule_teams):
        if name in stats_teams:
            rows.append({"schedule_name": name, "stats_name": name})

    # 2) mapping manuel supplémentaire
    manual_map = {
        # Clubs déjà présents
        "Alavés": "Alaves",
        "Almería": "Almeria",
        "Cádiz": "Cadiz",
        "Leganés": "Leganes",
        "Köln": "Koln",
        "Eint Frankfurt": "Eintracht Frankfurt",
        "Gladbach": "Monchengladbach",
        "Leverkusen": "Bayer Leverkusen",
        "Betis": "Real Betis",
        "Manchester Utd": "Manchester United",
        "Newcastle Utd": "Newcastle United",
        "Nott'ham Forest": "Nottingham Forest",
        "Paris S-G": "Paris Saint Germain",
        "Saint-Étienne": "Saint Etienne",
        "St. Pauli": "St Pauli",
        "Tottenham": "Tottenham Hotspur",
        "West Ham": "West Ham United",
        "Wolves": "Wolverhampton Wanderers",

        
        "Atlético Madrid": "Atletico Madrid",
        "Brighton": "Brighton and Hove Albion",
        "Inter": "Internazionale",
        "Sheffield Utd": "Sheffield United",

        # Équipes absentes des team_stats FBRef (Bundesliga 2 par ex.)
        "Düsseldorf": "",
        "Elversberg": "",

        # Sélections nationales
        "Albania": "Albania Men",
        "Austria": "Austria Men",
        "Belgium": "Belgium Men",
        "Croatia": "Croatia Men",
        "Czechia": "Czechia Men",
        "Denmark": "Denmark Men",
        "England": "England Men",
        "France": "France Men",
        "Georgia": "Georgia Men",
        "Germany": "Germany Men",
        "Hungary": "Hungary Men",
        "Italy": "Italy Men",
        "Netherlands": "Netherlands Men",
        "Poland": "Poland Men",
        "Portugal": "Portugal Men",
        "Romania": "Romania Men",
        "Scotland": "Scotland Men",
        "Serbia": "Serbia Men",
        "Slovakia": "Slovakia Men",
        "Slovenia": "Slovenia Men",
        "Spain": "Spain Men",
        "Switzerland": "Switzerland Men",
        "Türkiye": "Turkiye Men",
        "Ukraine": "Ukraine Men",
    }

    # Ajouter les mappings manuels
    for sched_name, stats_name in manual_map.items():
        if sched_name in schedule_teams:
            rows.append({"schedule_name": sched_name, "stats_name": stats_name})

    # 3) Ajouter le reste (si présent dans team_stats)
    already_mapped = {r["schedule_name"] for r in rows}

    for name in sorted(schedule_teams):
        if name not in already_mapped:
            rows.append({
                "schedule_name": name,
                "stats_name": name if name in stats_teams else ""
            })

    # Sauvegarde
    mapping_df = pd.DataFrame(rows).drop_duplicates(subset=["schedule_name"], keep="first")

    os.makedirs("data", exist_ok=True)
    mapping_df.to_csv(out_path, index=False)

    log(f"Saved mapping → {out_path}")

    unmapped = mapping_df[mapping_df["stats_name"] == ""]
    log("Unmapped rows (stats_name empty):")
    if not unmapped.empty:
        print(unmapped.to_string(index=False))
    else:
        print("All schedule teams have a mapping.")


if __name__ == "__main__":
    main()
