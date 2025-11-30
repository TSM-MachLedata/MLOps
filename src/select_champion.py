import json
import os
from pathlib import Path
from datetime import datetime


METRICS_FILES = {
    "model1": Path("reports/model1_metrics.json"),
    "model2": Path("reports/model2_metrics.json"),
    "model3_player_mode": Path("reports/model3_player_mode_metrics.json"),
}

# ordre d'importance des métriques pour le ranking
METRIC_PRIORITY = ["f1_macro", "accuracy"]


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


def main():
    log("🔎 Reading metrics for model comparison...")

    metrics_by_model = {}
    for name, path in METRICS_FILES.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                metrics_by_model[name] = json.load(f)
        else:
            log(f"⚠️ Metrics file missing for {name}: {path}")

    if not metrics_by_model:
        raise SystemExit("❌ No metrics files found, cannot select champion.")

    def score(model_name: str):
        m = metrics_by_model[model_name]
        # tuple (f1_macro, accuracy)
        return tuple(float(m.get(k, 0.0)) for k in METRIC_PRIORITY)

    # argmax sur le tuple (f1, acc)
    champion_key = max(metrics_by_model.keys(), key=score)

    log(f"🏆 Champion model selected: {champion_key}")
    log(f"   Metrics: {metrics_by_model[champion_key]}")

    config = {
        "champion_key": champion_key,
        "metric_priority": METRIC_PRIORITY,
        "metrics": metrics_by_model[champion_key],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs("app", exist_ok=True)
    champion_path = Path("app/champion_config.json")
    with champion_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log(f"💾 Champion config written to {champion_path}")


if __name__ == "__main__":
    main()
