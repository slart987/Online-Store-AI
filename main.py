from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize app
app = Flask(__name__)

# Load trained models and label classes
cat_pipeline = joblib.load("cat_pipeline.joblib")
num_pipeline = joblib.load("num_pipeline.joblib")
y_cat_classes = joblib.load("y_cat_classes.joblib")

# These are needed for inverse transforming categorical predictions
from sklearn.preprocessing import LabelEncoder
CAT_TARGETS = ["Brand", "Model", "CPU", "GPU", "OS", "Color"]
NUM_TARGETS = ["RAM (GB)", "Storage (GB)", "Screen (in)",
               "Weight (kg)", "Battery Life (hrs)", "Release Year"]

# Reconstruct label encoders
y_encoders = {}
for col, classes in y_cat_classes.items():
    le = LabelEncoder()
    le.classes_ = classes
    y_encoders[col] = le

# Prediction class (same as in notebook)
class LaptopAttributePredictor:
    def __init__(self, cat_pipe, num_pipe, y_label_decoders):
        self.cat_pipe = cat_pipe
        self.num_pipe = num_pipe
        self.decoders = y_label_decoders

    def predict(self, purpose: str, price: float) -> dict:
        Xnew = pd.DataFrame([{"Purpose": purpose, "Price ($)": price}])
        cat_pred_enc = self.cat_pipe.predict(Xnew)[0]
        cat_pred = {
            col: self.decoders[col].inverse_transform([enc])[0]
            for col, enc in zip(CAT_TARGETS, cat_pred_enc)
        }
        num_pred = {col: float(val) for col, val
                    in zip(NUM_TARGETS, self.num_pipe.predict(Xnew)[0])}
        return {**cat_pred, **num_pred}

# Instantiate the predictor
predictor = LaptopAttributePredictor(cat_pipeline, num_pipeline, y_encoders)

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        purpose = data["Purpose"]
        price = float(data["Price ($)"])
        result = predictor.predict(purpose, price)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
