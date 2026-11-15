
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "ml"))

from predict import predict

app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "demo.html")


@app.route("/predict", methods=["POST"])
def predict_review():
    try:
        data = request.get_json()

        label, probability, features = predict(data)

        return jsonify({
            "prediction": label,
            "suspicious_probability": float(probability),
            "features": features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
