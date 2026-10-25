"""Build Final_DataSet.csv from the original Sleep_dataset.csv.

1. Add three behavioural columns (Caffeine Intake, Daily Work Hours, Screen Time) to the real rows.
2. Generate synthetic records for quality scores 1-9 (the original data only covers 4-9).
3. Concatenate both and flag every row with `Is Synthetic`.

The run is seeded, so the same file is produced every time.
"""
import numpy as np
import pandas as pd

import config

N_SYNTHETIC = 24_300
NOISE_LEVEL = 0.04

# (low, high) per quality score; ranges overlap on purpose so neighbouring scores are not trivially separable.
BASE_RANGES = {
    1: {"Sleep Duration": (3.0, 4.0), "PA": (0, 30), "Stress": (7.5, 9.5), "Steps": (1000, 2500)},
    2: {"Sleep Duration": (3.5, 4.5), "PA": (20, 50), "Stress": (6.5, 8.5), "Steps": (2000, 3500)},
    3: {"Sleep Duration": (4.0, 5.0), "PA": (40, 70), "Stress": (5.5, 7.5), "Steps": (3000, 4500)},
    4: {"Sleep Duration": (4.5, 5.5), "PA": (60, 90), "Stress": (4.5, 6.5), "Steps": (4000, 5500)},
    5: {"Sleep Duration": (5.0, 6.0), "PA": (80, 110), "Stress": (3.5, 5.5), "Steps": (5000, 6500)},
    6: {"Sleep Duration": (5.5, 6.8), "PA": (90, 130), "Stress": (3.0, 5.0), "Steps": (6000, 8000)},
    7: {"Sleep Duration": (6.0, 7.2), "PA": (100, 140), "Stress": (2.5, 4.5), "Steps": (7000, 9000)},
    8: {"Sleep Duration": (6.5, 7.5), "PA": (110, 150), "Stress": (2.0, 4.0), "Steps": (8000, 10000)},
    9: {"Sleep Duration": (7.0, 8.0), "PA": (120, 160), "Stress": (1.5, 3.5), "Steps": (9000, 11000)},
}
INTEGER_COLUMNS = ["Physical Activity Level", "Daily Steps", "Heart Rate"]
NOISY_COLUMNS = ["Sleep Duration", "Physical Activity Level", "Stress Level", "Daily Steps",
                 "Caffeine Intake", "Daily Work Hours", "Screen Time", "Heart Rate"]
COPIED_COLUMNS = ["Person ID", "Gender", "Age", "Occupation", "BMI Category", "Blood Pressure"]


# ---------------------------------------------------------------- behavioural columns (real rows)
def assign_caffeine(row, rng):
    if row["Occupation"] in ("Doctor", "Nurse"):
        return rng.integers(150, 401) if row["Stress Level"] > 6 else rng.integers(100, 301)
    if row["Occupation"] in ("Software Engineer", "Engineer"):
        return rng.integers(100, 351) if row["Age"] < 30 else rng.integers(50, 251)
    return rng.integers(0, 201)


def assign_work_hours(row, rng):
    if row["Occupation"] in ("Doctor", "Nurse"):
        return rng.uniform(8, 12)
    if row["Occupation"] in ("Software Engineer", "Engineer"):
        return rng.uniform(6, 10)
    if row["Occupation"] in ("Teacher", "Accountant"):
        return rng.uniform(6, 8)
    return rng.uniform(7, 9)


def assign_screen_time(row, rng):
    if row["Occupation"] in ("Software Engineer", "Engineer"):
        return rng.uniform(5, 10) if row["Age"] < 30 else rng.uniform(3, 8)
    return rng.uniform(1, 6) if row["Age"] < 40 else rng.uniform(1, 5)


def add_behavioural_features(df, rng):
    df = df.copy()
    df["Caffeine Intake"] = df.apply(assign_caffeine, axis=1, rng=rng)
    df["Daily Work Hours"] = df.apply(assign_work_hours, axis=1, rng=rng).round(2)
    df["Screen Time"] = df.apply(assign_screen_time, axis=1, rng=rng).round(2)
    return df


# ---------------------------------------------------------------- synthetic records
def generate_record(quality, real, rng, noise_level=NOISE_LEVEL):
    r = BASE_RANGES[quality]
    rec = {
        "Sleep Duration": rng.uniform(*r["Sleep Duration"]),
        "Physical Activity Level": rng.integers(r["PA"][0], r["PA"][1] + 1),
        "Stress Level": rng.uniform(*r["Stress"]),
        "Daily Steps": rng.integers(r["Steps"][0], r["Steps"][1] + 1),
        "Caffeine Intake": rng.uniform(max(0, 500 - quality * 50), max(50, 500 - quality * 30)),
        "Daily Work Hours": rng.uniform(max(4, 12 - quality), max(8, 12 - quality / 1.5)),
        "Screen Time": rng.uniform(max(0, 10 - quality), max(6, 10 - quality)),
        "Heart Rate": rng.integers(max(50, 100 - quality * 5), max(75, 110 - quality * 3) + 1),
    }
    options = ["None", "Insomnia", "Sleep Apnea"] if quality <= 3 else ["None", "Insomnia"]
    rec["Sleep Disorder"] = str(rng.choice(options))

    for col in NOISY_COLUMNS:
        value = rec[col] + rng.normal(0, noise_level * rec[col])
        rec[col] = max(0, int(value)) if col in INTEGER_COLUMNS else round(max(0, value), 2)
    rec["Stress Level"] = min(rec["Stress Level"], 10.0)

    donor = real.iloc[rng.integers(len(real))]
    for col in COPIED_COLUMNS:
        rec[col] = donor[col]
    rec[config.TARGET] = quality
    return rec


def build_dataset(seed=config.SEED, n_synthetic=N_SYNTHETIC):
    rng = np.random.default_rng(seed)
    real = pd.read_csv(config.RAW_DATA_PATH, keep_default_na=False)
    real["Sleep Disorder"] = real["Sleep Disorder"].replace("", "None")
    real = add_behavioural_features(real, rng)

    qualities = list(range(1, 10))
    labels = [qualities[i % len(qualities)] for i in range(n_synthetic)]
    synthetic = pd.DataFrame([generate_record(q, real, rng) for q in labels])
    synthetic = synthetic[real.columns].sample(frac=1, random_state=seed).reset_index(drop=True)

    real[config.SYNTHETIC_FLAG] = 0
    synthetic[config.SYNTHETIC_FLAG] = 1
    return pd.concat([real, synthetic], ignore_index=True)


def main():
    final = build_dataset()
    config.DATA_DIR.mkdir(exist_ok=True)
    final.to_csv(config.FINAL_DATA_PATH, index=False)
    print(f"Saved {config.FINAL_DATA_PATH.name}: {final.shape}")
    print(final.groupby(config.SYNTHETIC_FLAG)[config.TARGET].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
