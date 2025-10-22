import streamlit as st
import pandas as pd
import joblib
import warnings
import os

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- The FULL Advice Database ---
ADVICE_DATABASE = {
    "Software Engineer": {
        "Stress Level": "Your role requires intense focus. To manage stress, use techniques like the Pomodoro method to schedule regular breaks away from the screen, allowing your mind to reset.",
        "Physical Activity Level": "A sedentary role can impact sleep. Counter this by incorporating short walks or stretching sessions every hour to boost circulation and energy.",
        "Sleep Duration": "Avoid complex problem-solving or coding right before bed. This 'mental cooldown' period is crucial for allowing your brain to switch off and prepare for sleep.",
        "Daily Steps": "Hitting a daily step goal is important to offset long hours of sitting. Try a walk during your lunch break or after work to reach at least 7,000-8,000 steps.",
        "Heart Rate": "A high resting heart rate can be linked to mental stress. Regular cardio exercise is the most effective way to lower it over time.",
        "BMI Category": "For a sedentary job, maintaining a healthy weight is key. Combining regular activity with mindful eating can prevent weight gain and its negative effects on sleep."
    },
    "Doctor": {
        "Stress Level": "The high-pressure environment of your job is a major factor. Short mindfulness or breathing exercises between patient consultations can significantly lower acute stress.",
        "Sleep Duration": "Irregular hours and on-call shifts disrupt your circadian rhythm. On your days off, prioritize a consistent sleep schedule to help your body recover and reset.",
        "Physical Activity Level": "Even a 20-minute walk can help you decompress after a long shift and improve sleep quality.",
        "Daily Steps": "Your days are busy, but tracking steps can reveal how much you truly move. Aim for consistency, even if the total number isn't high every day.",
        "Heart Rate": "Your resting heart rate is a key indicator of recovery. If it's elevated, it's a sign your body needs more rest to cope with the demands of your job.",
        "BMI Category": "Managing weight can be challenging with a demanding schedule. Even small dietary improvements and consistent activity can reduce the risk of sleep apnea and improve restfulness."
    },
    "Nurse": {
        "Stress Level": "Your work is both physically and emotionally draining. It is essential to dedicate time to a personal, relaxing activity after each shift to decompress and separate from work.",
        "Sleep Duration": "If you work rotating shifts, creating an optimal sleep environment is key. Use blackout curtains, a white noise machine, or earplugs to ensure uninterrupted rest.",
        "Physical Activity Level": "While you're on your feet all day, this is often low-intensity stress. A dedicated session of moderate exercise (like a brisk walk or jog) can help relieve tension and improve sleep depth.",
        "Daily Steps": "You likely take many steps at work, but a walk outdoors provides different benefits, like sun exposure and mental relaxation, which also aid sleep.",
        "Heart Rate": "Pay attention to your heart rate as a sign of physical and emotional fatigue. If it's consistently high, it's a signal to prioritize your own rest and recovery.",
        "BMI Category": "Maintaining a healthy weight is important for handling the physical demands of your job and ensuring good sleep for recovery."
    },
    "Lawyer": {
        "Stress Level": "The high-stakes nature of your work can lead to chronic stress. Establishing a firm boundary between work life and personal time is the most critical step for long-term well-being.",
        "Sleep Duration": "While working late is sometimes unavoidable, consistently sacrificing sleep for billable hours will ultimately reduce your cognitive performance and sharpness.",
        "Physical Activity Level": "Use physical activity as a tool to clear your mind. A workout can improve focus for complex legal work and help you sleep better.",
        "Daily Steps": "Long hours at a desk are common. Make a conscious effort to get up and walk around frequently. Aiming for a step goal can help.",
        "Heart Rate": "Mental stress directly impacts heart rate. Techniques like meditation can help manage your physiological response to stress.",
        "BMI Category": "Long hours at a desk can lead to weight gain. Prioritizing a healthy diet and scheduling time for exercise is essential for both physical health and sleep."
    },
    "Teacher": {
        "Stress Level": "Constant classroom engagement requires significant mental energy. Finding a few minutes of quiet solitude between classes or during your lunch break is vital to manage stress levels.",
        "Sleep Duration": "Grading and lesson planning can easily extend into your sleep time. Set a strict 'work cutoff' time in the evening to protect your sleep window.",
        "Physical Activity Level": "Physical activity can be a great outlet. Consider an after-school routine to burn off stress.",
        "Daily Steps": "Being on your feet in the classroom contributes to your step count, but a dedicated walk can provide additional stress-reducing benefits.",
        "Heart Rate": "Managing a classroom can be stressful. Practice deep-breathing exercises to help keep your heart rate calm during challenging moments.",
        "BMI Category": "Maintaining a healthy weight helps with the energy levels needed for teaching. Prioritize healthy meals, even on busy school days."
    },
    "Accountant": {"Stress Level": "Tax season and end-of-quarter deadlines are high-stress periods. Proactively schedule short breaks and relaxation activities during these times to prevent burnout.", "Physical Activity Level": "Long hours at a desk are common. Ensure you get up and move regularly, and consider a post-work exercise routine to decompress.", "Sleep Duration": "Avoid working on complex spreadsheets right before bed. Give your brain time to switch off.", "Daily Steps": "A sedentary job means you need to be intentional about steps. Taking the stairs or a brisk walk at lunch can make a big difference.", "Heart Rate": "Notice if your heart rate is higher during peak stress periods. This is a sign to step away and take a short break.", "BMI Category": "Mindful eating during long workdays is crucial. Plan healthy snacks to avoid unhealthy choices driven by stress."},
    "Sales Representative": {"Stress Level": "The pressure of meeting targets can be a significant source of stress. Focus on process-oriented goals rather than just outcomes to manage this pressure.", "Sleep Duration": "Travel and irregular schedules can disrupt sleep. Try to maintain a consistent wind-down routine in the evening, regardless of your location.", "Physical Activity Level": "Use hotel gyms or go for a walk to explore a new city. Staying active on the road is key for sleep.", "Daily Steps": "Travel can be sedentary. Make an effort to walk through airports instead of sitting, and aim for a step goal each day.", "Heart Rate": "Stress from sales calls can elevate heart rate. Practice calming techniques before important meetings.", "BMI Category": "Eating out while traveling can impact your BMI. Look for healthier options on menus and stay hydrated."},
    "Manager": {"Stress Level": "Managing a team comes with its own set of pressures. Delegating effectively and protecting your personal time are key strategies for stress reduction.", "Physical Activity Level": "Back-to-back meetings can lead to a sedentary day. Try scheduling 'walking meetings' or taking calls while walking to increase activity.", "Sleep Duration": "It's hard to switch off from people-management issues. Journaling or writing down a to-do list for the next day can help clear your mind before bed.", "Daily Steps": "Lead by example. Encourage walking breaks for your team and yourself.", "Heart Rate": "If you feel your heart rate rising in a difficult meeting, focus on your breathing to stay calm and centered.", "BMI Category": "Stress-eating can be a challenge. Keep healthy snacks at your desk to make better choices during a busy day."},
    "Engineer": {"Stress Level": "Complex problem-solving can be mentally taxing. Ensure you disconnect from work-related thoughts in the evening to allow for mental rest.", "Sleep Duration": "Late-night project work can impact sleep. A consistent 'lights-out' time is crucial for maintaining your internal clock.", "Physical Activity Level": "Similar to software engineers, your role can be sedentary. Make movement a regular part of your day.", "Daily Steps": "Set a timer to remind yourself to get up and walk around the office or site every hour.", "Heart Rate": "Frustration from a difficult problem can raise your heart rate. Step away for a few minutes to clear your head.", "BMI Category": "Focus on a balanced diet to fuel your brain for complex tasks, which also helps in maintaining a healthy weight."},
    "Scientist": {"Stress Level": "The pressure of research and experiments can be high. Ensure you have hobbies and activities outside of work to decompress.", "Sleep Duration": "A well-rested mind is more creative and analytical. Protect your sleep as you would a critical experiment.", "Physical Activity Level": "Long hours in the lab can be either sedentary or physically demanding. Tailor your exercise to balance your workday.", "Daily Steps": "Whether at a desk or in a lab, make sure you're getting enough general movement throughout the day.", "Heart Rate": "Curiosity and discovery can be exciting, but manage the associated stress. A calm heart rate supports a calm mind.", "BMI Category": "Consistent meal times, even during long experiments, can help regulate your metabolism and maintain a healthy BMI."},
    "Salesperson": {"Stress Level": "Facing rejection and pressure to perform can be very stressful. Develop resilience techniques and focus on the activities you can control.", "Sleep Duration": "A good night's sleep is crucial for the energy and positive attitude needed in sales. Make it a non-negotiable priority.", "Physical Activity Level": "Exercise is a great way to burn off the stress of a challenging day with clients.", "Daily Steps": "If your job involves being on your feet, you may have a high step count. If not, make walking a priority.", "Heart Rate": "Practice calming techniques before a big sales pitch to manage performance anxiety.", "BMI Category": "Healthy eating provides sustained energy for a demanding sales role. Avoid relying on caffeine and sugar."},
    "default": { 
        "Stress Level": "High stress is a primary disruptor of deep sleep. Consider incorporating daily relaxation techniques like meditation, deep breathing, or journaling.",
        "Sleep Duration": "A consistent sleep schedule is the foundation of good sleep. Aim to go to bed and wake up at the same time, even on weekends.",
        "Physical Activity Level": "Regular physical activity is proven to improve sleep quality. Even a 30-minute brisk walk in the afternoon can make a significant difference.",
        "Daily Steps": "A higher daily step count is strongly linked to better overall health and improved sleep. Aim for at least 7,000 steps a day.",
        "Heart Rate": "Your resting heart rate is a key indicator of your cardiovascular health. Improving fitness through regular exercise is the best way to support a healthy heart rate.",
        "BMI Category": "Maintaining a healthy weight is one of the most effective ways to improve sleep quality and reduce the risk of sleep-related breathing disorders."
    }
}

class SleepConsultant:
    def __init__(self, pipeline_path):
        try:
            pipeline = joblib.load(pipeline_path)
            self.model = pipeline["model"]
            self.scaler = pipeline["scaler"]
            self.label_encoders = pipeline["label_encoders"]
            self.model_columns = pipeline["model_columns"]
        except FileNotFoundError:
            st.error(f"FATAL ERROR: The model pipeline file was not found at the specified path. The application cannot start.")
            st.stop()
    
    def _predict_quality(self, data_dict):
        df = pd.DataFrame([data_dict.copy()])
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop(['Blood Pressure'], axis=1)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try: df.loc[:, col] = encoder.transform(df[col])
                except ValueError: df.loc[:, col] = -1
        df_aligned = df.reindex(columns=self.model_columns, fill_value=0)
        df_aligned[self.scaler.get_feature_names_out()] = self.scaler.transform(df_aligned[self.scaler.get_feature_names_out()])
        return self.model.predict(df_aligned)[0]

    def generate_report(self, user_profile):
        """
        Generates a full, detailed report as a formatted Markdown string.
        """
        report_lines = []

        # --- Phase 1: Heavily Expanded and Detailed Analysis of User's Current State ---
        report_lines.append("## Detailed Analysis of Your Current Profile")
        duration = user_profile.get('Sleep Duration', 0)
        stress = user_profile.get('Stress Level', 0)
        activity = user_profile.get('Physical Activity Level', 0)
        hr = user_profile.get('Heart Rate', 0)
        steps = user_profile.get('Daily Steps', 0)
        bmi_category = user_profile.get('BMI Category', 'Normal')

        # In-depth analysis for Sleep Duration
        if duration < 6.0: report_lines.append(f"- **Sleep Duration:** At `{duration:.1f}` hours, your sleep is critically low. This is likely the primary factor affecting your energy, mood, and cognitive function.")
        elif duration < 7.0: report_lines.append(f"- **Sleep Duration:** Your `{duration:.1f}` hours is below the recommended 7-9 hours. Even a small increase could lead to noticeable improvements.")
        else: report_lines.append(f"- **Sleep Duration:** At `{duration:.1f}` hours, your sleep duration is within the optimal healthy range. This is a strong foundation.")

        # In-depth analysis for Stress Level
        if stress >= 8: report_lines.append(f"- **Stress Level:** A score of `{stress}/10` is very high and is a major obstacle to restorative sleep. High cortisol levels from stress can prevent you from reaching deep sleep stages.")
        elif stress >= 6: report_lines.append(f"- **Stress Level:** A score of `{stress}/10` is high and is likely impacting your sleep quality, even if you are getting enough hours.")
        else: report_lines.append(f"- **Stress Level:** Your stress level of `{stress}/10` is in a manageable range. This is a positive factor for your sleep.")
        
        # In-depth analysis for Physical Activity
        if activity < 30: report_lines.append(f"- **Physical Activity:** `{activity}` minutes of daily activity is low and represents a significant opportunity for improvement. Regular exercise is a powerful tool to increase sleep pressure.")
        elif activity < 60: report_lines.append(f"- **Physical Activity:** `{activity}` minutes is a good start. Increasing this towards 60 minutes of moderate activity daily could further enhance your sleep depth.")
        else: report_lines.append(f"- **Physical Activity:** `{activity}` minutes of daily activity is an excellent amount that strongly supports high-quality sleep.")

        # In-depth analysis for Daily Steps
        if steps < 5000: report_lines.append(f"- **Daily Steps:** A count of `{steps}` indicates a largely sedentary lifestyle, which can negatively affect sleep patterns and overall health.")
        elif steps < 7500: report_lines.append(f"- **Daily Steps:** Your count of `{steps}` is a good baseline. Pushing this towards the 8,000-10,000 range can lead to better health and sleep outcomes.")
        else: report_lines.append(f"- **Daily Steps:** Your step count of `{steps}` is excellent and reflects an active lifestyle that promotes good sleep.")

        # In-depth analysis for Heart Rate
        if hr > 85: report_lines.append(f"- **Heart Rate:** Your resting heart rate of `{hr}` bpm is significantly elevated. This is often a sign of high stress or poor cardiovascular fitness, both of which are detrimental to sleep.")
        elif hr > 75: report_lines.append(f"- **Heart Rate:** Your resting heart rate of `{hr}` bpm is on the higher side of normal. This could be an indicator of underlying stress or a need for more cardiovascular exercise.")
        else: report_lines.append(f"- **Heart Rate:** Your resting heart rate of `{hr}` bpm is in a healthy, optimal range.")
        
        # In-depth analysis for BMI Category
        if bmi_category in ['Overweight', 'Obese']: report_lines.append(f"- **BMI Category:** Your category of '{bmi_category}' indicates excess body weight, which is a significant risk factor for poor sleep quality and sleep apnea.")
        else: report_lines.append(f"- **BMI Category:** Your category of '{bmi_category}' is healthy and supports good sleep quality.")

        # --- Phase 2: The Prediction ---
        predicted_score = self._predict_quality(user_profile)
        report_lines.append("---")
        report_lines.append(f"### Based on this detailed analysis, your predicted sleep quality score is: **{predicted_score:.1f} / 10**")

        # --- Phase 3: The Optimization Plan ---
        report_lines.append("## Your Personalized Optimization Plan")
        if predicted_score >= 8.5:
            report_lines.append("Your predicted score is in an excellent range. Your current habits are creating a strong foundation for quality sleep. Maintain this consistency.")
            return "\n".join(report_lines)
            
        simulations = [
            {"desc": "Increasing your sleep duration", "feat": "Sleep Duration", "cond": lambda v: v < 8.0, "change": 0.5},
            {"desc": "Reducing your daily stress level", "feat": "Stress Level", "cond": lambda v: v > 3, "change": -1},
            {"desc": "Incorporating more physical activity", "feat": "Physical Activity Level", "cond": lambda v: v < 75, "change": 30}
        ]
        results = []
        for sim in simulations:
            if sim["cond"](user_profile.get(sim["feat"], 0)):
                sim_data = user_profile.copy()
                sim_data[sim["feat"]] += sim["change"]
                new_score = self._predict_quality(sim_data)
                if new_score - predicted_score > 0.1: results.append({"desc": sim["desc"], "new_score": new_score})
        
        if results:
            results.sort(key=lambda x: x['new_score'], reverse=True)
            best_rec = results[0]
            report_lines.append(f"### Primary Recommendation (Highest Impact):")
            report_lines.append(f"- Our simulation shows that **'{best_rec['desc']}'** is the most effective change for you.")
            report_lines.append(f"- **Predicted Outcome:** This single change could boost your score from `{predicted_score:.1f}` to approximately `{best_rec['new_score']:.1f}`.")
        
        report_lines.append("### Detailed Action Plan:")
        occupation = user_profile.get('Occupation', 'default')
        if duration < 7.0: report_lines.append(f"- **To Increase Sleep Duration:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('Sleep Duration')}")
        if stress >= 7: report_lines.append(f"- **To Manage Stress:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('Stress Level')}")
        if activity < 45: report_lines.append(f"- **To Increase Physical Activity:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('Physical Activity Level')}")
        if steps < 7000: report_lines.append(f"- **To Increase Daily Steps:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('Daily Steps')}")
        if hr > 80: report_lines.append(f"- **To Improve Heart Rate:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('Heart Rate')}")
        if bmi_category in ['Overweight', 'Obese']: report_lines.append(f"- **To Improve BMI:** {ADVICE_DATABASE.get(occupation, ADVICE_DATABASE['default']).get('BMI Category')}")

        return "\n".join(report_lines)

# ==============================================================================
# --- Streamlit Application UI ---
# ==============================================================================

# Use caching to load the model only once
@st.cache_resource
def load_consultant(pipeline_path):
    # Check if the file exists before attempting to load
    if not os.path.exists(pipeline_path):
        st.error(f"FATAL ERROR: The model pipeline file was not found at the specified path: {pipeline_path}")
        st.error("Please ensure the file exists and the path is correct.")
        st.stop()
    return SleepConsultant(pipeline_path)

def main():
    st.set_page_config(page_title="Sleep Consultant", layout="wide")
    st.title("Personalized Sleep Quality Consultant 😴")
    st.write("Enter your daily metrics, and our AI will provide a detailed analysis and a personalized action plan to improve your sleep quality.")

    # --- Load the model ---
    PIPELINE_PATH = r"C:\Users\Laptop World\Desktop\SleepingQualityPredictionOptimization\Model\sleep_pipeline.pkl"
    consultant = load_consultant(PIPELINE_PATH)

    # --- User Inputs in the Sidebar ---
    st.sidebar.header("Enter Your Details:")
    
    # Define lists for select boxes
    occupation_list = sorted([key for key in ADVICE_DATABASE.keys() if key != 'default'])
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    occupation = st.sidebar.selectbox("Occupation", occupation_list)
    bmi_category = st.sidebar.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])
    sleep_disorder = st.sidebar.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])

    age = st.sidebar.slider("Age", 25, 60, 40)
    sleep_duration = st.sidebar.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
    activity_level = st.sidebar.slider("Physical Activity Level (minutes/day)", 0, 120, 45)
    stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 6)
    heart_rate = st.sidebar.slider("Resting Heart Rate (bpm)", 50, 100, 75)
    daily_steps = st.sidebar.slider("Daily Steps", 1000, 15000, 6000)
    bp = st.sidebar.text_input("Blood Pressure (e.g., 120/80)", "120/80")

    # --- Generate Report Button ---
    if st.sidebar.button("Generate My Sleep Report"):
        
        # Basic validation for blood pressure format
        if '/' not in bp or len(bp.split('/')) != 2:
            st.error("Invalid Blood Pressure format. Please use the format 'Systolic/Diastolic' (e.g., '120/80').")
        else:
            user_profile = {
                'Gender': gender, 'Age': age, 'Occupation': occupation,
                'Sleep Duration': sleep_duration, 'Physical Activity Level': activity_level,
                'Stress Level': stress_level, 'BMI Category': bmi_category,
                'Blood Pressure': bp, 'Heart Rate': heart_rate,
                'Daily Steps': daily_steps, 'Sleep Disorder': sleep_disorder
            }
            
            with st.spinner('Analyzing your profile and generating personalized recommendations...'):
                report = consultant.generate_report(user_profile)
                st.markdown(report)

if __name__ == "__main__":
    main()