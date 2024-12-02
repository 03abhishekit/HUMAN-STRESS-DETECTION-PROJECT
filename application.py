from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('Model/best_stress_level_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result=""

    if request.method == 'POST':
       # Collect input data from the form
       input_data = [float(request.form[feature]) for feature in request.form]
    
        # Scale the input data using the same scaler as used during training
       input_array = np.array([input_data])
       input_scaled = scaler.transform(input_array)
    
       # Predict stress level
       prediction = model.predict(input_scaled)[0]

       # Map prediction to stress level
       stress_levels = {0: "Low Stress", 1: "Moderate Stress", 2: "High Stress"}
       stress_level = stress_levels.get(int(prediction), "Unknown")  # Convert to regular int

       # Render the result.html template and pass the prediction and stress level
       return render_template('result.html', stress_level=stress_level, prediction=int(prediction))

    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)
