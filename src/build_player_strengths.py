import pandas as pd
from datetime import datetime

def log(msg):
    print(f"[[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]] {msg}")

def to_float(series):
    """Convertit une série en float, remplace les valeurs non-numériques par 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)

def safe_get(df, col):
    """Retourne la colonne si elle existe sinon retourne une série de 0."""
    if col in df.columns:
        return to_float(df[col])
    print(f"⚠️ Missing column: {col}, using zeros.")
    return pd.Series([0] * len(df))

def main():
    log("Loading player stats...")
    df = pd.read_csv("data/raw/player_season_stats_model2.csv")

    log("Extracting meaningful features...")

    # === EXTRACTION SÉCURISÉE DES STATS FBREF ===
    df["goals"]        = safe_get(df, "Performance")       # Gls
    df["assists"]      = safe_get(df, "Performance.1")     # Ast
    df["yellow_cards"] = safe_get(df, "Performance.6")     # CrdY
    df["red_cards"]    = safe_get(df, "Performance.7")     # CrdR

    df["xG"]           = safe_get(df, "Expected")          # xG
    df["xAG"]          = safe_get(df, "Expected.2")        # xAG
    df["minutes"]      = safe_get(df, "Playing Time.2")    # Min

    df["goals_per90"]   = safe_get(df, "Per 90 Minutes")      # Gls/90
    df["assists_per90"] = safe_get(df, "Per 90 Minutes.1")    # Ast/90
    df["xG_per90"]      = safe_get(df, "Per 90 Minutes.5")    # xG/90

    # === CALCUL SCORE JOUEUR ===
    df["player_score"] = (
        df["goals"] * 4 +
        df["assists"] * 3 +
        df["xG"] * 1.5 +
        df["xAG"] * 1.2 +
        df["goals_per90"] * 2 +
        df["assists_per90"] * 1.5 +
        df["xG_per90"] * 1 -
        df["yellow_cards"] * 0.5 -
        df["red_cards"] * 2
    )

    # Nettoyage : garder seulement joueurs valides
    df = df[df["team"].notna() & df["player"].notna()]

    # === FORCE : aucun score = 0 systématique n'est possible maintenant ===
    df["player_score"] = df["player_score"].fillna(0)

    # === MOYENNE PAR ÉQUIPE ===
    log("Computing team strengths...")

    team_strengths = (
        df.groupby("team")["player_score"]
        .mean()
        .reset_index()
        .rename(columns={"player_score": "team_strength"})
    )

    # === SAUVEGARDE ===
    df.to_csv("data/processed/player_strengths.csv", index=False)
    team_strengths.to_csv("data/processed/team_strengths.csv", index=False)

    log("Saved → data/processed/player_strengths.csv")
    log("Saved → data/processed/team_strengths.csv")

if __name__ == "__main__":
    main()
