"""Model loading, input validation and the "what-if" simulation used by the web app."""
import json

import joblib
import numpy as np
import pandas as pd

import config
from src.preprocessing import encode_sleep_disorder

# (label, feature, change, lower bound, upper bound)
WHAT_IF_LEVERS = [
    ("Sleep 1 hour longer", "Sleep Duration", +1.0, None, 9.0),
    ("Cut stress by 2 points", "Stress Level", -2.0, 1.0, None),
    ("Cut caffeine by 100 mg", "Caffeine Intake", -100.0, 0.0, None),
    ("Cut screen time by 2 hours", "Screen Time", -2.0, 0.0, None),
    ("Add 20 minutes of activity", "Physical Activity Level", +20.0, None, None),
    ("Walk 2,000 more steps", "Daily Steps", +2000.0, None, None),
    ("Work 1 hour less", "Daily Work Hours", -1.0, 3.0, None),
]


class SleepPredictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.stats = json.loads(config.STATS_PATH.read_text())

    def clean_input(self, data):
        """Fill missing numeric fields with dataset means and validate the sleep disorder."""
        user = dict(data)
        for key in config.NUMERIC_FEATURES:
            raw = data.get(key)
            if raw in (None, ""):
                user[key] = self.stats[key]["mean"]
            else:
                try:
                    user[key] = float(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"'{key}' must be a number.")
        try:
            user["Weight"] = float(data.get("Weight") or config.DEFAULT_WEIGHT_KG)
        except (TypeError, ValueError):
            raise ValueError("'Weight' must be a number.")
        user["Sleep Disorder"] = data.get("Sleep Disorder") or "None"
        encode_sleep_disorder(user["Sleep Disorder"])  # raises on unknown values
        return user

    def score(self, user):
        """Probability-weighted average of the predicted quality classes (a smooth 1-9 score)."""
        row = {k: user[k] for k in config.FEATURES}
        row["Sleep Disorder"] = encode_sleep_disorder(row["Sleep Disorder"])
        proba = self.model.predict_proba(pd.DataFrame([row], columns=config.FEATURES))[0]
        return float(np.dot(proba, self.model.classes_))

    def what_if(self, user, base_score=None, top=3):
        """Score each single lifestyle change and return the most helpful ones."""
        base = self.score(user) if base_score is None else base_score
        results = []
        for label, feature, delta, low, high in WHAT_IF_LEVERS:
            changed = dict(user)
            value = user[feature] + delta
            if low is not None:
                value = max(low, value)
            if high is not None:
                value = min(high, value)
            if value == user[feature]:
                continue
            changed[feature] = value
            new_score = self.score(changed)
            results.append({"action": label, "new_score": round(new_score, 2), "gain": round(new_score - base, 2)})
        results.sort(key=lambda r: r["gain"], reverse=True)
        return [r for r in results if r["gain"] > 0][:top]
