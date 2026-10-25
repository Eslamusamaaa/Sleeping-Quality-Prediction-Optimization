"""Flask app: serves the front-end and the /predict API.

Run from the project root:  python BackEnd/app.py   ->  http://127.0.0.1:5000
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory

import config
from src.advice import analysis_paragraph, optimization_paragraph
from src.predictor import SleepPredictor

FRONTEND_DIR = config.ROOT / "FrontEnd"

app = Flask(__name__, static_folder=None)

try:
    predictor = SleepPredictor()
except FileNotFoundError as exc:
    raise SystemExit(f"{exc}\nTrain the model first:  python run_pipeline.py") from exc


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON payload"}), 400
    try:
        user = predictor.clean_input(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    score = predictor.score(user)
    # show 3000 instead of 3000.0 in the generated text
    shown = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in user.items()}
    return jsonify({
        "score": round(score, 1),
        "current": analysis_paragraph(shown, score),
        "plan": optimization_paragraph(shown, score),
        "what_if": predictor.what_if(user, base_score=score),
    })


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
