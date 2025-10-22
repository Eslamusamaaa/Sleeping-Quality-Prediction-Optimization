# Personalized Sleep Quality Consultant

An AI-powered web application that analyzes your daily habits and provides a detailed, personalized consultation to help you improve your sleep quality. Built with Python, Scikit-learn, and Streamlit.

---

## Project Overview

This project addresses the common problem of poor sleep quality by moving beyond generic advice. It uses a machine learning model (RandomForest Regressor) trained on a comprehensive sleep dataset to predict a user's sleep quality score based on their lifestyle inputs.

More importantly, it acts as a **virtual sleep consultant**, providing a detailed analysis of the user's current habits and a multi-point, customized action plan with occupation-specific advice.

---

## Key Features

-   **Detailed Habit Analysis:** Get an in-depth breakdown of your key metrics (Sleep Duration, Stress, Activity, Heart Rate, BMI) and how they compare to healthy benchmarks.
-   **AI-Powered Prediction:** Receive a sleep quality score from 1-10, predicted by a trained machine learning model.
-   **"What-If" Simulation:** The system simulates the impact of small lifestyle changes to identify the single most effective recommendation for you.
-   **Customized Action Plan:** Receive a detailed, multi-point plan with actionable advice tailored to your specific areas of improvement and your occupation.
-   **Interactive UI:** A simple and intuitive web interface built with Streamlit allows for easy input and clear visualization of the results.

---

## Tech Stack

-   **Backend & ML:** Python
-   **Data Manipulation:** Pandas
-   **Machine Learning Model:** Scikit-learn (RandomForestRegressor)
-   **Web Framework / UI:** Streamlit
-   **Model Persistence:** Joblib

---

## How to Use

Follow these steps to run the application on your local machine.

### Prerequisites

-   Python 3.8+
-   pip package manager
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Eslamusamaaa/Sleeping-Quality-Prediction-Optimization.git](https://github.com/Eslamusamaaa/Sleeping-Quality-Prediction-Optimization.git)
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd Sleeping-Quality-Prediction-Optimization
    ```

3.  **Install the required dependencies:**
    This project's dependencies are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the Model File:**
    Ensure your trained model file, `sleep_pipeline.pkl`, is located in the `Model` directory as specified in the code.

5.  **Run the Streamlit Application:**
    Navigate to the `UI` directory and run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

The application should now be running and accessible in your web browser.

---
