from typing import List, Optional, Dict

import json
import os
import traceback  # NEW

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print(">>> [app.main] import starting")
print(">>> [app.main] CWD:", os.getcwd())
print(">>> [app.main] ls .:", os.listdir("."))

critical_paths = [
    "data/processed/model2_training_dataset.csv",
    "data/processed/player_strengths.csv",
    "data/raw/team_stats_multi_leagues.csv",
    "data/raw/team_match_stats_model2.csv",
    "models/model2_xgb.json",
    "app/models/home_model.json",
    "app/models/away_model.json",
]
for p in critical_paths:
    print(f">>> [app.main] exists? {p} -> {os.path.exists(p)}")

app = FastAPI(
    title="Football Prediction MLOps API",
    description=(
        "Serve multi-league regression (model1), team-level classifier (model2), "
        "and player-mode classifier (model3)."
    ),
    version="1.2.0",
)

# -------------------------------------------------
# Champion config
# -------------------------------------------------

CHAMPION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "champion_config.json")
CHAMPION_KEY = os.getenv("CHAMPION_KEY_OVERRIDE")  # optional override via env

champion_config = {}
try:
    print(f">>> [app.main] Checking champion config at {CHAMPION_CONFIG_PATH}")
    if os.path.exists(CHAMPION_CONFIG_PATH):
        with open(CHAMPION_CONFIG_PATH, "r", encoding="utf-8") as f:
            champion_config = json.load(f)
        if not CHAMPION_KEY:
            CHAMPION_KEY = champion_config.get("champion_key", "model2")
    else:
        CHAMPION_KEY = CHAMPION_KEY or "model2"
except Exception as e:
    print("⚠️ [app.main] Error loading champion_config.json:", e)
    traceback.print_exc()
    CHAMPION_KEY = CHAMPION_KEY or "model2"

print(f"🏆 Loaded champion key: {CHAMPION_KEY}")

def with_champion_tag(base_key: str) -> str:
    return base_key + (" (champion)" if CHAMPION_KEY == base_key else "")


# -------------------------------------------------
# Artifacts
# -------------------------------------------------

model2_clf: Optional[xgb.XGBClassifier] = None
df_model2_train: Optional[pd.DataFrame] = None
MODEL2_TEAMS: List[str] = []

player_strengths_df: Optional[pd.DataFrame] = None
team_stats_df: Optional[pd.DataFrame] = None
match_stats_df: Optional[pd.DataFrame] = None

home_reg: Optional[xgb.XGBRegressor] = None
away_reg: Optional[xgb.XGBRegressor] = None

PLAYER_TEAMS: List[str] = []
MODEL1_TEAMS: List[str] = []  # based on team_stats_multi_leagues.csv

MODEL1_TEAM_LOOKUP: Dict[str, str] = {}
MODEL2_TEAM_LOOKUP: Dict[str, str] = {}
PLAYER_TEAM_LOOKUP: Dict[str, str] = {}


def _build_lookup(names: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for n in names:
        if n is None:
            continue
        canon = str(n).strip()
        if not canon:
            continue
        key = canon.lower()
        lookup.setdefault(key, canon)
    return lookup


# ---- Model 2 artifacts ------------------------------------------------
try:
    print(">>> [app.main] Loading model2_xgb.json & training dataset...")
    model2_clf = xgb.XGBClassifier()
    model2_clf.load_model("models/model2_xgb.json")
    df_model2_train = pd.read_csv("data/processed/model2_training_dataset.csv")

    if "home_team_clean" in df_model2_train.columns and "away_team_clean" in df_model2_train.columns:
        MODEL2_TEAMS = sorted(
            set(df_model2_train["home_team_clean"].dropna().unique())
            | set(df_model2_train["away_team_clean"].dropna().unique())
        )
    MODEL2_TEAM_LOOKUP = _build_lookup(MODEL2_TEAMS)
    print(f"✅ Loaded model2 classifier & training dataset. {len(MODEL2_TEAMS)} teams.")
except Exception as e:
    print("⚠️ Could not load model2 artifacts:", e)
    traceback.print_exc()


# ---- Player-mode + team stats artifacts (model3 + model1) ------------
try:
    print(">>> [app.main] Loading player_strengths & team_stats...")
    player_strengths_df = pd.read_csv("data/processed/player_strengths.csv")
    team_stats_df = pd.read_csv("data/raw/team_stats_multi_leagues.csv")
    match_stats_df = pd.read_csv("data/raw/team_match_stats_model2.csv")

    if player_strengths_df is not None:
        player_strengths_df["team_norm"] = (
            player_strengths_df["team"].astype(str).str.strip().str.lower()
        )
        player_strengths_df["player_norm"] = (
            player_strengths_df["player"].astype(str).str.strip().str.lower()
        )

    if "team" in player_strengths_df.columns:
        PLAYER_TEAMS = sorted(player_strengths_df["team"].dropna().unique())
        PLAYER_TEAM_LOOKUP = _build_lookup(PLAYER_TEAMS)

    if "team" in team_stats_df.columns:
        MODEL1_TEAMS = sorted(team_stats_df["team"].dropna().unique())
        MODEL1_TEAM_LOOKUP = _build_lookup(MODEL1_TEAMS)

    if match_stats_df is not None and "opponent" in match_stats_df.columns:
        match_stats_df["opponent_norm"] = (
            match_stats_df["opponent"].astype(str).str.strip().str.lower()
        )

    print(
        f"✅ Loaded player-mode artifacts. "
        f"{len(PLAYER_TEAMS)} teams with player strengths, "
        f"{len(MODEL1_TEAMS)} teams with long-term stats."
    )
except Exception as e:
    print("⚠️ Could not load player-mode / team-stats artifacts:", e)
    traceback.print_exc()


# ---- Model 1 regression (home/away goals) -----------------------------
try:
    print(">>> [app.main] Loading model1 home/away regressors...")
    home_reg = xgb.XGBRegressor()
    away_reg = xgb.XGBRegressor()
    home_reg.load_model("app/models/home_model.json")
    away_reg.load_model("app/models/away_model.json")
    print("✅ Loaded model1 regression models.")
except Exception as e:
    print("⚠️ Could not load model1 regression models:", e)
    traceback.print_exc()


# -------------------------------------------------
# Pydantic schemas
# -------------------------------------------------

class TeamRequest(BaseModel):
    home_team: str
    away_team: str


class PlayerMatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_players: List[str]
    away_players: List[str]


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def validate_team_exists(team: str, context: str = "model2") -> str:
    raw = (team or "").strip()
    key = raw.lower()

    if context == "model2":
        canonical = MODEL2_TEAM_LOOKUP.get(key)
        where = "model2"
    elif context == "players":
        canonical = PLAYER_TEAM_LOOKUP.get(key)
        where = "player strengths"
    elif context == "model1":
        canonical = MODEL1_TEAM_LOOKUP.get(key)
        where = "model1"
    else:
        canonical = None
        where = context

    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown team '{team}' for {where}. Team must exist in the training data."),
        )

    return canonical

# ... 👇 keep the rest of your file (build_features_model1, build_features_model2,
# compute_team_strength, build_features_player_mode, and the FastAPI endpoints)
# exactly as you already have them; no changes needed down there.
