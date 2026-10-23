"""Train on complete synthetic trips; tune baseline on validation trips only."""
import json
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from config import DATA, MODELS, FEATURES
from processing.features import extract


def dataset(ids):
    x, y = [], []
    for i in ids:
        df = pd.read_csv(DATA / "scenarios" / f"trip_{i:02d}.csv")
        for _, part in df.groupby("window_id"):
            x.append(extract(part))
            y.append(part.label.iloc[0])
    return pd.DataFrame(x), y


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    x, y = dataset(manifest["splits"]["train"])
    xv, yv = dataset(manifest["splits"]["validation"])
    thresholds = [v/10 for v in range(10,81)]
    threshold = max(thresholds, key=lambda t: f1_score([v=="pothole" for v in yv], xv.peak>t, zero_division=0))
    MODELS.mkdir(exist_ok=True)
    for name, features in [("motion", FEATURES), ("speed", FEATURES+["speed_kmh"])]:
        model = RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_leaf=2, random_state=42, n_jobs=1)
        model.fit(x[features], y)
        with (MODELS/f"{name}.pkl").open("wb") as f:
            pickle.dump({"model": model, "features": features}, f)
    (MODELS/"metadata.json").write_text(json.dumps({"source": "synthetic", "threshold": threshold,
        "train_trips": manifest["splits"]["train"], "validation_trips": manifest["splits"]["validation"],
        "note": "Uncalibrated model probabilities; not road-validated."}, indent=2))
    print(f"Models trained; validation-selected threshold: {threshold} m/s²")

if __name__ == "__main__":
    main()
