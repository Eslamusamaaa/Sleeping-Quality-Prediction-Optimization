"""Loading, cleaning and encoding helpers shared by training, EDA and the web app."""
import pandas as pd

import config


def load_final_dataset():
    df = pd.read_csv(config.FINAL_DATA_PATH, keep_default_na=False)
    df["Sleep Disorder"] = df["Sleep Disorder"].replace("", "None")
    return df


def encode_sleep_disorder(value):
    """Map a disorder name to the integer the model was trained on. Unknown names raise ValueError."""
    text = str(value).strip().lower()
    for code, name in enumerate(config.SLEEP_DISORDERS):
        if text == name.lower():
            return code
    raise ValueError(f"Unsupported sleep disorder '{value}'. Choose one of {config.SLEEP_DISORDERS}.")


def build_features(df):
    X = df[config.FEATURES].copy()
    X["Sleep Disorder"] = X["Sleep Disorder"].map(encode_sleep_disorder)
    return X


def split_real_synthetic(df):
    """Return (real_unique, synthetic). The original data repeats rows, so real rows are deduplicated."""
    real = df[df[config.SYNTHETIC_FLAG] == 0].drop_duplicates(subset=config.REAL_ROW_KEY)
    synthetic = df[df[config.SYNTHETIC_FLAG] == 1]
    return real, synthetic
