# 🌿 Plant Disease Detector

An end-to-end Deep Learning project that classifies plant leaf images into **38 distinct categories** (covering 14 species) to detect health status and specific diseases. Leveraging **MobileNetV2** with Transfer Learning in **PyTorch**, this detector achieves **~96% validation accuracy** and exposes a lightweight **Flask REST API** for real-time inference.

---

## 🚀 Key Features

* **38 Classes Classified**: Identifies healthy leaves as well as specific bacterial, fungal, and viral diseases across 14 crops.
* **State-of-the-Art Architecture**: Uses a pre-trained **MobileNetV2** backbone for high efficiency, lightweight model footprint, and fast CPU/GPU inference.
* **REST API Endpoint**: Flask application exposing a `/predict` endpoint to process and return classifications in structured JSON.
* **Data Augmentations**: Incorporates custom brightness, contrast, saturation, and horizontal flip transforms to prevent overfitting.
* **Interactive Development**: Structured Jupyter Notebooks for training, evaluation, validation, and real-time visualization.

---

## 📁 Repository Structure

```text
├── models/
│   ├── classes.pkl                   # Pickled list of the 38 classification classes
│   └── mobilenet_v2_model.pt         # Saved weights for the trained MobileNetV2 model
├── api.py                            # Flask REST API server exposing /predict
├── helpers.py                        # Utility functions for class formatting and category indexing
├── models.py                         # PyTorch model architecture definitions (MobileNetV2, TinyVGG)
├── transforms.py                     # Data augmentation and evaluation pipeline transformations
├── model_development.ipynb           # Jupyter Notebook detailing preprocessing, training, and validation
├── test.ipynb                        # Jupyter Notebook for testing and manual model verification
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation (this file)
```

---

## 📊 Dataset & Supported Classes

The model is trained on the popular **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)**. It classifies images into 38 labels representing the following plant-disease pairs:

| Plant Species | Supported Target Classes |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Haunglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted spider mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```sh
git clone https://github.com/aami-sagnik/plant-disease-detector.git
cd plant-disease-detector
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries:
```sh
pip install torch torchvision flask numpy pillow matplotlib pandas torchmetrics torchinfo
```

---

## 🖥️ Running the REST API

To deploy the Flask server locally for inference, run:
```sh
python api.py
```
The server will start at `http://127.0.0.1:5000`.

### Sending a Prediction Request
You can send POST requests containing a leaf image file to the `/predict` endpoint:

```sh
curl -X POST \
  -F "file=@/path/to/your/plant_image.jpg" \
  http://127.0.0.1:5000/predict
```

### Example JSON Responses

* **Diseased Leaf Detected:**
  ```json
  {
    "disease": "early blight",
    "is_healthy": false,
    "plant": "potato",
    "probability": 0.98452
  }
  ```

* **Healthy Leaf Detected:**
  ```json
  {
    "disease": null,
    "is_healthy": true,
    "plant": "tomato",
    "probability": 0.99824
  }
  ```

---

## 🧠 Model Architecture & Training

* **Base Extractor**: MobileNetV2 with pre-trained weights (`MobileNet_V2_Weights.DEFAULT`).
* **Feature Freezing**: Extractor layers are frozen (`requires_grad = False`) to speed up convergence and leverage general features.
* **Classifier Head**: Replaced with a fully connected layer mapped to `38` output nodes.
* **Loss Function**: Cross-Entropy Loss.
* **Data Prep**: Images are resized to `224x224` and converted to PyTorch tensors. Training data goes through horizontal flips and color jittering.
* **Performance**: Achieves **~95.79% accuracy** on validation data within just a few epochs of classifier fine-tuning.

