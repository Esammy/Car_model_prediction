from flask import Flask, request, jsonify
from ultralytics import YOLO
# import sympy
import numpy as np

model = YOLO('./weights/last.pt')


app = Flask(__name__)

# Optionally, you can set the app to run in debug mode during development
app.config['DEBUG'] = True

def predict_cars(input_data):
    result = model(input_data)
    names_dict = result[0].names
    prob_score = result[0].probs.tolist()
    pred_car = names_dict[np.argmax(prob_score)]

    return {"car_make_model": pred_car}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (assuming JSON input)
        data = request.json

        # Call your prediction model to get the result
        prediction_result = predict_cars(data)

        # Return the prediction result as a JSON response
        return jsonify(prediction_result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()