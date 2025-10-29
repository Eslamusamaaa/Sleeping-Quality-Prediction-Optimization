# Personalized Sleep Quality Consultant

An **AI-powered web application** that analyzes your daily habits and provides a detailed, personalized consultation to help you improve your sleep quality.  
Built with **Python**, **Scikit-learn**, **Flask**, and a custom **HTML/CSS/JavaScript** front-end.

---

## Project Overview

This project goes beyond generic sleep advice.  
It uses a **machine learning model (RandomForest Classifier)** trained on an enriched sleep dataset to predict a user's sleep quality score (1–10) based on their lifestyle inputs.

More importantly, it acts as a **virtual sleep consultant**, providing a detailed analysis of the user's current habits and a **multi-point, customized action plan** with occupation-specific advice.

---

## Key Features

- **Detailed Habit Analysis:**  
  Get an in-depth breakdown of your key metrics (*Sleep Duration, Stress, Activity, Heart Rate, BMI, Caffeine, Work Hours, Screen Time*) and how they compare to healthy benchmarks.

- **AI-Powered Prediction:**  
  Receive a sleep quality score (1–10) predicted by a trained **Random Forest Classifier**.

- **“What-If” Simulation:**  
  The system simulates the impact of small lifestyle changes to identify the **single most effective improvement** for you.

- **Customized Action Plan:**  
  Receive a detailed, multi-point plan with actionable advice tailored to your specific areas of improvement **and your occupation**.

- **Interactive UI:**  
  A modern, responsive web interface built with **HTML, CSS, and JavaScript** for clear visualization and user-friendly interaction.

---

## Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend & ML** | Python, Flask |
| **Data Manipulation** | Pandas |
| **Machine Learning Model** | Scikit-learn (RandomForestClassifier) |
| **Frontend** | HTML, CSS, JavaScript |
| **Model Persistence** | Joblib |

---

## How to Use

Follow these steps to run the application on your local machine:

### Prerequisites

- Python **3.8+**  
- pip package manager  
- Git  

---

### Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Eslamusamaaa/Sleeping-Quality-Prediction-Optimization.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Sleeping-Quality-Prediction-Optimization
   ```

3. **Install dependencies:**

   The required packages are listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Place the Model File:**

   Ensure your trained model file `sleep_pipeline.pkl` is located in the `Model` directory as specified in the code.

5. **Run the Application:**

   ```bash
   python app.py
   ```

6. **Access the App:**

   Once the server starts, open your browser and go to:

   ```
   http://127.0.0.1:5000
   ```

---
