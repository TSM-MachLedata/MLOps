import pandas as pd
import os

def main():
    print(" Début du prétraitement des données multi-ligues...")

    # Définir les chemins
    raw_path = "data/raw"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)

    # Charger les fichiers
    schedule_file = os.path.join(raw_path, "schedule_multi_leagues.csv")
    team_stats_file = os.path.join(raw_path, "team_stats_multi_leagues.csv")

    schedule = pd.read_csv(schedule_file)
    team_stats = pd.read_csv(team_stats_file)

    print(f" Fichiers chargés : {len(schedule)} matchs, {len(team_stats)} lignes de stats")

    # Nettoyage et formatage du calendrier
    schedule = schedule.dropna(subset=["homeScore", "awayScore"])  # garder uniquement les matchs joués
    schedule["homeScore"] = pd.to_numeric(schedule["homeScore"], errors="coerce")
    schedule["awayScore"] = pd.to_numeric(schedule["awayScore"], errors="coerce")

    # Uniformiser les noms de colonnes
    schedule = schedule.rename(columns={
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "homeScore": "home_goals",
        "awayScore": "away_goals"
    })

    # Nettoyer la colonne date
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce", dayfirst=True)
    schedule = schedule.dropna(subset=["date"])

    # Ajouter le résultat du match
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "Home Win"
        elif row["home_goals"] < row["away_goals"]:
            return "Away Win"
        else:
            return "Draw"

    schedule["result"] = schedule.apply(get_result, axis=1)

    # Nettoyage du dataset des équipes
    team_stats = team_stats.rename(columns={"team": "team_name"})
    team_stats["goals_diff"] = team_stats["goals_for"] - team_stats["goals_against"]

    # Supprimer les doublons éventuels
    schedule = schedule.drop_duplicates(subset=["date", "home_team", "away_team"])

    # Fusion : ajouter les stats des équipes domicile / extérieur
    home_stats = team_stats.add_prefix("home_")
    away_stats = team_stats.add_prefix("away_")

    merged = schedule.merge(
        home_stats,
        left_on=["home_team", "league", "season"],
        right_on=["home_team_name", "home_league", "home_season"],
        how="left"
    ).merge(
        away_stats,
        left_on=["away_team", "league", "season"],
        right_on=["away_team_name", "away_league", "away_season"],
        how="left"
    )

    # Nettoyage final
    merged = merged.dropna(subset=["home_goals", "away_goals"])
    merged = merged.sort_values("date").reset_index(drop=True)

    print(f" Données fusionnées : {merged.shape[0]} matchs, {merged.shape[1]} colonnes")

    # Sauvegarder le fichier propre
    output_file = os.path.join(processed_path, "clean_matches.csv")
    merged.to_csv(output_file, index=False)

    print(f" Fichier final enregistré : {output_file}")
    print(" Prétraitement terminé avec succès !")

if __name__ == "__main__":
    main()
