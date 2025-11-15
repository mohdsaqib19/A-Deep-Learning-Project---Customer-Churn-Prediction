import flask
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = flask.Flask(__name__)

# --- Load Model and Preprocessor ---
try:
    model = load_model('churn_model.keras')
    print("* Model loaded successfully.")
except Exception as e:
    print(f"* Error loading model: {e}")
    model = None

# Load the fitted preprocessor
try:
    preprocessor = joblib.load('preprocessor.joblib')
    print("* Preprocessor loaded successfully.")
except Exception as e:
    print(f"* Error loading preprocessor: {e}")
    preprocessor = None

# --- Define Routes ---

# Main route to serve the index.html file
@app.route('/')
def home():
    return flask.render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return flask.jsonify({'error': 'Model or preprocessor not loaded.'}), 500

    try:
        # Get data from the POST request
        data = flask.request.json
        print(f"Received data: {data}")

        input_df = pd.DataFrame([data])
        
        # Log the DataFrame columns and dtypes for debugging
        print(f"Input DataFrame:\n{input_df.to_string()}")
        print(f"DataFrame dtypes:\n{input_df.dtypes}")

        # --- Preprocess the input data ---
        
        try:
             # Manually cast numeric features
            input_df['tenure'] = pd.to_numeric(input_df['tenure'], errors='coerce')
            input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce')
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        except Exception as e:
            print(f"Error casting types: {e}")
            return flask.jsonify({'error': f'Invalid numeric data: {e}'}), 400

        print(f"DataFrame dtypes after casting:\n{input_df.dtypes}")
        
        processed_input = preprocessor.transform(input_df)
        print("Data preprocessed successfully.")

        # --- Make Prediction ---
        prediction_prob = model.predict(processed_input)
        
        # The output is a probability (0 to 1)
        prob = float(prediction_prob[0][0])
        print(f"Raw prediction: {prob}")

        # Convert probability to a percentage and a simple "Yes/No"
        churn_probability_percent = prob * 100
        churn_decision = "Yes" if prob > 0.5 else "No" # 0.5 is the threshold

        
        return flask.jsonify({
            'churn_probability': churn_probability_percent,
            'churn_decision': churn_decision
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return flask.jsonify({'error': str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    
    from flask_cors import CORS
    CORS(app) 

    app.run(debug=True, port=5000)