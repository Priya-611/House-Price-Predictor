from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and encoders
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
MAIN_ROAD_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'main_road.pkl')
GUEST_ROOM_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'guest_room.pkl')
BASEMENT_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'basement.pkl')
AIR_CONDITIONING_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'air_conditioning.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(MAIN_ROAD_ENCODER_PATH, 'rb') as f:
    main_road_encoder = pickle.load(f)
with open(GUEST_ROOM_ENCODER_PATH, 'rb') as f:
    guest_room_encoder = pickle.load(f)
with open(BASEMENT_ENCODER_PATH, 'rb') as f:
    basement_encoder = pickle.load(f)
with open(AIR_CONDITIONING_ENCODER_PATH, 'rb') as f:
    air_conditioning_encoder = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

# API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        stories = int(data['stories'])
        mainroad = main_road_encoder.transform([data['mainroad']])[0]
        guestroom = guest_room_encoder.transform([data['guestroom']])[0]
        basement = basement_encoder.transform([data['basement']])[0]
        airconditioning = air_conditioning_encoder.transform([data['airconditioning']])[0]
        parking = int(data['parking'])
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, airconditioning, parking]])
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 

