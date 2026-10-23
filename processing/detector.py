import json
import math
import pickle
import pandas as pd
from config import MODELS
from processing.features import extract


class Detector:
    def __init__(self, method="speed"):
        if method not in {"threshold", "motion", "speed"}:
            raise ValueError("Unknown detector")
        self.method = method
        self.threshold = json.loads((MODELS/"metadata.json").read_text())["threshold"]
        if method != "threshold":
            # Only load locally generated artifacts, never user-uploaded pickle files.
            with (MODELS/f"{method}.pkl").open("rb") as f:
                self.bundle = pickle.load(f)

    def predict(self, frame):
        features = extract(frame)  # Labels and coordinates never enter inference.
        if self.method == "speed" and not math.isfinite(features["speed_kmh"]):
            raise ValueError("Speed-aware inference requires speed for every sample in this window.")
        if self.method == "threshold":
            label = "pothole" if features["peak"] > self.threshold else "non_pothole"
            confidence = None
        else:
            x = pd.DataFrame([features])[self.bundle["features"]]
            model = self.bundle["model"]
            probabilities = model.predict_proba(x)[0]
            index = probabilities.argmax()
            label, confidence = str(model.classes_[index]), float(probabilities[index])
        return {"label": label, "confidence": confidence, "features": features, "method": self.method}
