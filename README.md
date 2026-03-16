# Plant Disease Detector
Plant Disease Detector using the dataset https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset to achieve ~96% accuracy. The model uses MobileNetV2 as the architecture.

The test results are in `model_deployment.ipynb`.

To run the model for inference, simply run the following command at the root of this project.
```sh
python3 api.py
```
You can then send an image at the endpoint for prediction with the request below.
```sh
curl -X POST \                                     
  -F "file=@/path/to/file" \
  http://127.0.0.1:5000/predict
```
