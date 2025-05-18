from flask import Flask, request, jsonify
from models import MobileNetV2Model, output_shape
from transforms import eval_data_transform as transform
from helpers import get_is_healthy_plant_disease
import torch
import numpy as np
from PIL import Image, ImageFile
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MobileNetV2Model(output_shape).to(device)
model.load_state_dict(torch.load("./models/mobilenet_v2_model.pt"))
model.eval()

with open("./models/classes.pkl", "rb") as f:
    classes = pickle.load(f)

app = Flask(__name__)

def predict(img: ImageFile):
    img_tensor = transform(img).unsqueeze(dim=0).to(device)

    with torch.inference_mode():
        pred = model(img_tensor)
        pred = torch.softmax(pred, dim=1)
        proba = round(pred.max().item(), 5)
        pred = pred.argmax(dim=1).item()
        result = get_is_healthy_plant_disease(classes[pred])
        result["probability"] = proba
        return result

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['file']

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Not an image file'}), 400

    result = predict(img)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)



