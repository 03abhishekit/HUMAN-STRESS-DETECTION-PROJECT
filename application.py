from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler from pickle file
with open('Model/random_forest_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model and scaler from the dictionary
model = model_data['model']
scaler = model_data['scaler']

print("Model and scaler loaded successfully.")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_stress_level', methods=['GET', 'POST'])
def predict_stress_level():
    result = ""
    
    if request.method == 'POST':
            # Extract form values and convert them to float
            anxiety_level = float(request.form['anxiety_level'])
            self_esteem = float(request.form['self_esteem'])
            mental_health_history = float(request.form['mental_health_history'])
            depression = float(request.form['depression'])
            headache = float(request.form['headache'])
            blood_pressure = float(request.form['blood_pressure'])
            sleep_quality = float(request.form['sleep_quality'])
            breathing_problem = float(request.form['breathing_problem'])
            noise_level = float(request.form['noise_level'])
            living_conditions = float(request.form['living_conditions'])
            safety = float(request.form['safety'])
            basic_needs = float(request.form['basic_needs'])
            academic_performance = float(request.form['academic_performance'])
            study_load = float(request.form['study_load'])
            teacher_student_relationship = float(request.form['teacher_student_relationship'])
            future_career_concerns = float(request.form['future_career_concerns'])
            social_support = float(request.form['social_support'])
            peer_pressure = float(request.form['peer_pressure'])
            extracurricular_activities = float(request.form['extracurricular_activities'])
            bullying = float(request.form['bullying'])

            # Prepare the input data for prediction
            input_data = np.array([[anxiety_level, self_esteem, mental_health_history, depression, headache,
                                    blood_pressure, sleep_quality, breathing_problem, noise_level, living_conditions,
                                    safety, basic_needs, academic_performance, study_load, teacher_student_relationship,
                                    future_career_concerns, social_support, peer_pressure, extracurricular_activities, bullying]])

            # Scale the input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the loaded model
            prediction = model.predict(input_data_scaled)

            # Display result based on prediction
            if prediction[0] == 0:
                result = "The predicted stress level is: Low Level Stress"
            elif prediction[0] == 1:
                result = "The predicted stress level is: Mid Level Stress"
            else:
                result = "The predicted stress level is: High Level Stress"

        

        # If it's a GET request, just render the form or the input page
            return render_template('result.html', result=result)  # Render a form page where users can input data
    
    else:
        return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)



# # streamlit_app.py
# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model and scaler
# with open('Model/random_forest_model.pkl', 'rb') as file:
#     model_data = pickle.load(file)

# model = model_data['model']
# scaler = model_data['scaler']

# # Streamlit UI for input
# st.title("Stress Level Prediction")
# st.write("Enter your details to predict your stress level")

# # Input form
# anxiety_level = st.slider("Anxiety Level", min_value=0.0, max_value=10.0, step=0.1)
# self_esteem = st.slider("Self-Esteem", min_value=0.0, max_value=10.0, step=0.1)
# mental_health_history = st.slider("Mental Health History", min_value=0.0, max_value=10.0, step=0.1)
# depression = st.slider("Depression", min_value=0.0, max_value=10.0, step=0.1)
# headache = st.slider("Headache", min_value=0.0, max_value=10.0, step=0.1)
# blood_pressure = st.slider("Blood Pressure", min_value=0.0, max_value=10.0, step=0.1)
# sleep_quality = st.slider("Sleep Quality", min_value=0.0, max_value=10.0, step=0.1)
# breathing_problem = st.slider("Breathing Problem", min_value=0.0, max_value=10.0, step=0.1)
# noise_level = st.slider("Noise Level", min_value=0.0, max_value=10.0, step=0.1)
# living_conditions = st.slider("Living Conditions", min_value=0.0, max_value=10.0, step=0.1)
# safety = st.slider("Safety", min_value=0.0, max_value=10.0, step=0.1)
# basic_needs = st.slider("Basic Needs", min_value=0.0, max_value=10.0, step=0.1)
# academic_performance = st.slider("Academic Performance", min_value=0.0, max_value=10.0, step=0.1)
# study_load = st.slider("Study Load", min_value=0.0, max_value=10.0, step=0.1)
# teacher_student_relationship = st.slider("Teacher-Student Relationship", min_value=0.0, max_value=10.0, step=0.1)
# future_career_concerns = st.slider("Future Career Concerns", min_value=0.0, max_value=10.0, step=0.1)
# social_support = st.slider("Social Support", min_value=0.0, max_value=10.0, step=0.1)
# peer_pressure = st.slider("Peer Pressure", min_value=0.0, max_value=10.0, step=0.1)
# extracurricular_activities = st.slider("Extracurricular Activities", min_value=0.0, max_value=10.0, step=0.1)
# bullying = st.slider("Bullying", min_value=0.0, max_value=10.0, step=0.1)

# # Button to make the prediction
# if st.button("Predict Stress Level"):
#     # Prepare the input data
#     input_data = np.array([[anxiety_level, self_esteem, mental_health_history, depression, headache,
#                             blood_pressure, sleep_quality, breathing_problem, noise_level, living_conditions,
#                             safety, basic_needs, academic_performance, study_load, teacher_student_relationship,
#                             future_career_concerns, social_support, peer_pressure, extracurricular_activities, bullying]])

#     # Scale the input data using the loaded scaler
#     input_data_scaled = scaler.transform(input_data)

#     # Make prediction using the loaded model
#     prediction = model.predict(input_data_scaled)

#     # Display result based on prediction
#     if prediction[0] == 0:
#         result = "The predicted stress level is: Low Level Stress"
#     elif prediction[0] == 1:
#         result = "The predicted stress level is: Mid Level Stress"
#     else:
#         result = "The predicted stress level is: High Level Stress"

#     # Show the result
#     st.write(result)
