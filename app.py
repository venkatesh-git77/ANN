# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and feature columns
model = joblib.load('churn_model.pkl')
scaler = joblib.load('sc_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data['features']], columns=feature_columns[:len(data['features'])])
        print(input_df.columns())

        # Add missing columns if necessary
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0

        # Reorder columns to match training data order
        input_df = input_df[feature_columns]

        # Scale the input data
        scaled_data = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Return result as JSON
        result = {'prediction': int(prediction[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
