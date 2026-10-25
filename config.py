"""Central configuration: paths, feature lists and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "Model"
IMG_DIR = ROOT / "img" / "Sleeping-Quality-Prediction-Optimization"

RAW_DATA_PATH = DATA_DIR / "Sleep_dataset.csv"
FINAL_DATA_PATH = DATA_DIR / "Final_DataSet.csv"
MODEL_PATH = MODEL_DIR / "RandomForest_Model.pkl"
STATS_PATH = MODEL_DIR / "data_stats.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

SEED = 42
TARGET = "Quality of Sleep"
SYNTHETIC_FLAG = "Is Synthetic"

FEATURES = [
    "Screen Time", "Daily Work Hours", "Caffeine Intake", "Sleep Disorder",
    "Daily Steps", "Heart Rate", "Stress Level", "Physical Activity Level",
    "Sleep Duration", "Age",
]
NUMERIC_FEATURES = [f for f in FEATURES if f != "Sleep Disorder"]

# Order matters: the position in this list is the encoded value the model was trained on.
SLEEP_DISORDERS = ["Insomnia", "None", "Sleep Apnea"]

# Columns that identify one real record (used to drop duplicated real rows).
REAL_ROW_KEY = [
    "Gender", "Age", "Occupation", "Sleep Duration", TARGET, "Physical Activity Level",
    "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder",
]

# The dataset has no body measurements; the app uses these when the user does not give them.
DEFAULT_WEIGHT_KG = 78.0
DEFAULT_HEIGHT_CM = 170.0

# Real rows are ~1.5% of the data; they get more weight so the model does not only learn the generator.
REAL_SAMPLE_WEIGHT = 20
