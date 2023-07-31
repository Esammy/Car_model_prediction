from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np

model = YOLO('./weights/best.pt')


app = Flask(__name__)

# Optionally, you can set the app to run in debug mode during development
app.config['DEBUG'] = True


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (assuming JSON input)
        data = request.json

        # Call your prediction model to get the result
        result = model(data)
        names_dict = result[0].names
        prob = result[0].probs.tolist()
        pred_car = names_dict[np.argmax(prob)]



        # Return the prediction result as a JSON response
        return jsonify({'prediction': pred_car})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()