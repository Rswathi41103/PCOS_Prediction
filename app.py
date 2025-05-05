import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from data_utils import number_of_images_per_class, datafolder, preprocessing_image, train_model
import joblib
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Load the models (Ensure these files exist)
basic_model = joblib.load('basic_logistic_model.pkl')
advanced_model = joblib.load('hybrid_model.pkl')

def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # Resize to match model input
    # Add additional preprocessing steps if needed
    image = np.array(image) / 255.0  # Normalize the image
    return image

def preprocess_data(data):
    df = pd.DataFrame([data])
    # Apply any necessary preprocessing here
    return df.values


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about_1.html')

@app.route('/predict')
def predict():
    return render_template('predict_form.html')


@app.route('/irregular_periods')
def irregular():
    return render_template('irregular_periods.html')


@app.route('/hair_skin')
def hair_skin():
    return render_template('hair_skin.html')


@app.route('/androgens')
def androgens():
    return render_template('androgens.html')

@app.route('/ovarises')
def ovarises():
    return render_template('ovarises.html')

@app.route('/mood_changes')
def mood_changes():
    return render_template('changes.html')

@app.route('/weight')
def weight():
    return render_template('weight.html')


@app.route('/basic_calculation', methods=['GET'])
def basic_calculation():
    return render_template('basic_calculation.html')


@app.route('/advance_calculation', methods=['GET'])
def advance_calculation():
    return render_template('advance_calculation.html')

@app.route('/predict_basic', methods=['POST'])
def predict_basic():
    try:
        features = [float(request.form.get(feature, 0)) for feature in [
            'age', 'weight', 'height', 'bmi', 'Cycle(R/I)', 'pulse_rate', 'rr', 'pregnant',
            'weight_gain', 'hair_growth', 'skin_darkening', 'hair_loss',
            'pimples', 'fast_food', 'reg_exercise']]

        features = np.array(features).reshape(1, -1)

        prediction = basic_model.predict(features)
        prediction_proba = basic_model.predict_proba(features)

        result = {
            'prediction': 'Affected' if prediction[0] == 1 else 'Not Affected'
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    try:
        data = request.form.get('data')
        image = request.files.get('image')

        if not data or not image:
            return jsonify({'error': 'Data or image not provided'}), 400

        data = pd.read_json(data, typ='series').to_dict()
        image = image.read()

        image_features = preprocess_image(image)
        data_features = preprocess_data(data)

        image_features_flattened = image_features.flatten()
        data_features_flattened = data_features.flatten()

        combined_features = np.concatenate([image_features_flattened, data_features_flattened])

        prediction = advanced_model.predict([combined_features])

        result = 'Affected' if prediction[0] == 1 else 'Not affected'

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
