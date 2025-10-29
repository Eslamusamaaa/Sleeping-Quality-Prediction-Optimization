# -------------------------------------------------
# app.py  (enhanced version)
# -------------------------------------------------
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)                     # allow any origin (dev only)

# ---------------------------- Paths ------------------------------
RF_MODEL_PATH = r"C:\Users\Laptop World\Desktop\SleepingQualityPredictionOptimization\Model\RandomForest_Model.pkl"
LE_SLEEP_PATH = r"C:\Users\Laptop World\Desktop\SleepingQualityPredictionOptimization\Model\LabelEncoder_SleepDisorder.pkl"

# ---------------------------- Load -------------------------------
if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {RF_MODEL_PATH}")
if not os.path.exists(LE_SLEEP_PATH):
    raise FileNotFoundError(f"LabelEncoder not found: {LE_SLEEP_PATH}")

rf_model = joblib.load(RF_MODEL_PATH)
le_sleep = joblib.load(LE_SLEEP_PATH)

# ---------------------------- Stats ------------------------------
DATA_STATS = {
    "Screen Time": {"min": 0.9, "max": 10.1, "mean": 5.79},
    "Daily Work Hours": {"min": 3.4, "max": 12.9, "mean": 8.12},
    "Caffeine Intake": {"min": 1.0, "max": 525.0, "mean": 297.8},
    "Daily Steps": {"min": 900, "max": 12100, "mean": 5879},
    "Heart Rate": {"min": 50, "max": 119, "mean": 84.27},
    "Stress Level": {"min": 1.4, "max": 10.4, "mean": 5.06},
    "Physical Activity Level": {"min": 0, "max": 172, "mean": 85.18},
    "Sleep Duration": {"min": 2.7, "max": 9.0, "mean": 5.55},
    "Age": {"min": 27, "max": 59, "mean": 42.25},
    "Weight": {"min": 40, "max": 140, "mean": 78},
    "Height": {"mean": 170}                # needed for BMI (fallback)
}

# ---------------------------- Expected features -----------------
EXPECTED_FEATURES = [
    "Screen Time", "Daily Work Hours", "Caffeine Intake", "Sleep Disorder",
    "Daily Steps", "Heart Rate", "Stress Level", "Physical Activity Level",
    "Sleep Duration", "Age", "Weight", "Occupation", "Blood Pressure", "Gender"
]

# -------------------------------------------------
# Helper utilities
# -------------------------------------------------
def safe_get(data, key):
    if key in data and data[key] not in (None, ""):
        return data[key]
    stat = DATA_STATS.get(key)
    return stat["mean"] if stat else None

def parse_bp(bp_str):
    try:
        parts = str(bp_str).split('/')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None

def encode_sleep_disorder(val):
    try:
        return int(le_sleep.transform([str(val)])[0])
    except Exception:
        txt = str(val).strip().lower()
        mapping = {"none": "None", "no": "None", "n": "None", "0": "None",
                   "insomnia": "Insomnia", "apnea": "Sleep Apnea"}
        for k, v in mapping.items():
            if k in txt:
                return int(le_sleep.transform([v])[0])
        return 0

def bmi(weight_kg, height_cm=None):
    """Return BMI (float) – height optional (use dataset mean if missing)."""
    if height_cm is None:
        height_cm = DATA_STATS["Height"]["mean"]
    return round(weight_kg / ((height_cm/100)**2), 1)

# -------------------------------------------------
# Paragraph generators (unchanged analysis, **enhanced** plan)
# -------------------------------------------------
def analysis_paragraph(user, predicted_score):
    # (your original function – unchanged)
    sd = user["Sleep Duration"]
    stress = user["Stress Level"]
    steps = user["Daily Steps"]
    hr = user["Heart Rate"]
    pa = user["Physical Activity Level"]
    caffeine = user["Caffeine Intake"]
    work = user["Daily Work Hours"]
    occupation = user.get("Occupation", "your job")
    bp_sys, bp_dia = parse_bp(user.get("Blood Pressure", ""))
    gender = user.get("Gender", "")

    lines = []
    lines.append(f"Predicted sleep quality score is {predicted_score:.2f} out of 10 for the provided profile.")

    # Sleep duration
    if sd < 5.0:
        lines.append(f"Your sleep duration is low ({sd:.1f} hours), which is a strong contributor to poor sleep quality.")
    elif sd < 7.0:
        lines.append(f"Your sleep duration ({sd:.1f} hours) is below the recommended 7–9 hours and may limit recovery.")
    else:
        lines.append(f"Your sleep duration ({sd:.1f} hours) is within a healthy range and supports good recovery.")

    # Stress
    if stress >= 8:
        lines.append(f"Your stress level ({stress}/10) is very high and likely prevents restorative sleep cycles.")
    elif stress >= 6:
        lines.append(f"Your stress level ({stress}/10) is elevated and may worsen sleep fragmentation.")
    else:
        lines.append(f"Your stress level ({stress}/10) is within a manageable range.")

    # Activity & steps
    if pa < 30 or steps < 4000:
        lines.append(f"Physical activity is low (activity {pa} min/day, {steps} steps). Low daytime activity reduces sleep pressure.")
    elif pa >= 60 or steps >= 8000:
        lines.append(f"Your activity level (approx. {pa} minutes/day and {steps} steps) supports good sleep regulation.")
    else:
        lines.append(f"Your activity is moderate (about {pa} min/day and {steps} steps); small increases could help.")

    # Caffeine & heart rate
    if caffeine > 250:
        lines.append(f"Caffeine intake is high ({caffeine} mg/day) and may be elevating your heart rate and interfering with sleep onset.")
    elif caffeine > 100:
        lines.append(f"Caffeine intake ({caffeine} mg/day) is moderate; avoid taking it late in the day.")
    else:
        lines.append(f"Caffeine intake ({caffeine} mg/day) is low and unlikely to be a primary factor.")

    if hr > 95:
        lines.append(f"Resting heart rate is high ({hr} bpm), which may reflect stress or poor recovery.")
    elif hr > 80:
        lines.append(f"Resting heart rate ({hr} bpm) is slightly elevated; improving fitness or lowering stress can help.")
    else:
        lines.append(f"Resting heart rate ({hr} bpm) is in a healthy range.")

    # Work hours & occupation
    if work > 10:
        lines.append(f"Long work hours ({work} hrs/day) can reduce sleep opportunity and increase stress in {occupation}.")
    elif work < 5:
        lines.append(f"Short work hours ({work} hrs/day) give you opportunity to increase structured activity and sleep consistency.")
    else:
        lines.append(f"Work hours ({work} hrs/day) appear reasonable; prioritize a consistent bedtime.")

    # Blood pressure
    if bp_sys:
        if bp_sys >= 140 or bp_dia >= 90:
            lines.append(f"Blood pressure reading ({bp_sys}/{bp_dia}) is elevated and can affect sleep quality—consider medical follow-up.")
        else:
            lines.append(f"Blood pressure reading ({bp_sys}/{bp_dia}) is within standard ranges.")

    return " ".join(lines)

# -------------------------------------------------
# ENHANCED OPTIMIZATION PLAN
# -------------------------------------------------
def optimization_paragraph(user, predicted_score):
    sd   = user["Sleep Duration"]
    stress = user["Stress Level"]
    steps = user["Daily Steps"]
    hr   = user["Heart Rate"]
    pa   = user["Physical Activity Level"]
    caffeine = user["Caffeine Intake"]
    work = user["Daily Work Hours"]
    occupation = user.get("Occupation", "your job").lower()
    weight = user.get("Weight")
    gender = user.get("Gender", "").lower()
    bp_sys, bp_dia = parse_bp(user.get("Blood Pressure", ""))
    screen = user["Screen Time"]

    adv = []

    # ---------- Sleep Duration ----------
    if sd < 6:
        adv.append("Gradually add 30-60 min of sleep each week until you reach 7-8 h. Use a fixed bedtime & wake-up alarm.")
    elif sd < 7:
        adv.append("Shift bedtime 15-30 min earlier; dim lights & avoid screens 1 h before bed.")
    else:
        adv.append("Keep a rock-solid schedule – same bedtime & wake-up, even on weekends.")

    # ---------- Stress ----------
    if stress >= 8:
        adv.append("Practice 10-min deep-breathing twice daily + short walks during work breaks. Consider a mindfulness app.")
    elif stress >= 6:
        adv.append("Add a 15-min evening wind-down (progressive muscle relaxation, reading).")
    else:
        adv.append("Maintain current habits; use a 2-min breathing break on stressful days.")

    # ---------- Activity ----------
    if pa < 30 or steps < 4000:
        adv.append("Aim for ≥30 min moderate activity daily & 7-9 k steps. Stand/walk every hour.")
    elif pa < 60 or steps < 8000:
        adv.append("Target 45-60 min activity & 8-10 k steps most days to deepen sleep.")
    else:
        adv.append("You’re active! Add gentle stretching or a 10-min evening stroll for recovery.")

    # ---------- Caffeine ----------
    if caffeine > 300:
        adv.append("Cap caffeine at 150-200 mg/day; none after 2 pm. Switch last coffee to decaf.")
    elif caffeine > 150:
        adv.append("Consume caffeine only before noon; replace afternoon drinks with herbal tea.")
    else:
        adv.append("Keep caffeine <100 mg after lunch to preserve sleep latency.")

    # ---------- Heart Rate ----------
    if hr > 90:
        adv.append("Add low-intensity cardio (brisk walk, cycling) 3×/week; track HR weekly.")
    elif hr > 80:
        adv.append("Include 20-min cardio sessions 3×/week + daily relaxation.")
    else:
        adv.append("Maintain current routine; focus on sleep hygiene.")

    # ---------- Screen Time ----------
    if screen > 7:
        adv.append("Reduce evening screen time by 50 %; enable night-mode & use blue-light glasses.")
    elif screen > 5:
        adv.append("Limit screens 1 h before bed; replace with reading or light stretching.")
    else:
        adv.append("Your screen habits are good; keep devices out of the bedroom.")

    # ---------- Occupation-specific ----------
    occ_map = {
        "doctor": "Take strategic power-naps (≤20 min) on long shifts; protect off-day sleep.",
        "nurse": "Same as doctors; use blackout curtains for day-sleep.",
        "programmer": "Code early in the day; shut down screens 1 h before bed; use a “shutdown ritual”.",
        "engineer": "Same as programmers.",
        "teacher": "Finish grading/planning by 7 pm; reserve evenings for relaxation.",
        "student": "Study in focused 25-min blocks; avoid all-nighters.",
        "driver": "Maintain circadian rhythm with consistent sleep location & dark curtains.",
        "sales": "Delegate evening admin; set a hard stop at 8 pm.",
        "manager": "Same as sales."
    }
    for key, txt in occ_map.items():
        if key in occupation:
            adv.append(txt)
            break
    else:
        adv.append("Align sleep with work schedule; add morning sunlight & reduce evening screens.")

    # ---------- Weight / BMI ----------
    if weight:
        bmi_val = bmi(weight)
        if bmi_val > 30:
            adv.append(f"BMI {bmi_val} (obese). Gradual weight loss (0.5 kg/week) via diet + activity improves breathing & sleep.")
        elif bmi_val > 25:
            adv.append(f"BMI {bmi_val} (overweight). Combine daily activity with balanced meals to reach healthy range.")
        else:
            adv.append(f"BMI {bmi_val} – healthy. Maintain with consistent exercise & nutrition.")

    # ---------- Blood Pressure ----------
    if bp_sys and (bp_sys >= 140 or bp_dia >= 90):
        adv.append("BP is high. Consult a clinician; reduce sodium, increase potassium, and improve sleep.")

    # ---------- Gender nuance ----------
    if gender == "female":
        adv.append("Women often need slightly more sleep (7-9 h). Track menstrual cycle impact on sleep quality.")
    elif gender == "male":
        adv.append("Men benefit from post-workout protein timing to aid recovery & sleep.")

    # ---------- Quick-wins ----------
    adv.append("**Quick wins:** 1. No screens 1 h before bed. 2. 10-min walk after dinner. 3. Consistent wake-up time.")

    # Final motivational line
    adv.append("Small, consistent changes compound – pick 1-2 actions, track for a week, then add more.")

    return " ".join(adv)

# -------------------------------------------------
# /predict endpoint – returns JSON
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400

        # ----- build user dict with safe defaults -----
        user = {}
        for f in EXPECTED_FEATURES:
            user[f] = data.get(f) if f in data and data[f] not in (None, "") else (
                float(DATA_STATS[f]["mean"]) if f in DATA_STATS else ""
            )

        # ----- numeric conversion -----
        numeric_keys = ["Screen Time","Daily Work Hours","Caffeine Intake",
                        "Daily Steps","Heart Rate","Stress Level",
                        "Physical Activity Level","Sleep Duration","Age","Weight"]
        for k in numeric_keys:
            try:
                user[k] = float(user[k])
            except Exception:
                user[k] = float(DATA_STATS.get(k, {}).get("mean", 0))

        # ----- encode Sleep Disorder -----
        user["Sleep Disorder"] = encode_sleep_disorder(user.get("Sleep Disorder", "None"))

        # ----- model input -----
        model_features = ["Screen Time","Daily Work Hours","Caffeine Intake","Sleep Disorder",
                          "Daily Steps","Heart Rate","Stress Level","Physical Activity Level",
                          "Sleep Duration","Age"]
        X = pd.DataFrame([{k: user[k] for k in model_features}])
        pred = float(rf_model.predict(X)[0])

        # ----- paragraphs -----
        current = analysis_paragraph(user, pred)
        plan    = optimization_paragraph(user, pred)

        return jsonify({
            "score": round(pred, 2),
            "current": current,
            "plan": plan
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)