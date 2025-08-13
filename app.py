from flask import Flask, render_template, request
import joblib
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "models", "model.joblib")

app = Flask(__name__)

# Lazy-load the model so app starts even if model is missing (dev)
_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, proba=None, text="")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "").strip()
    if not text:
        return render_template("index.html", result="Please enter some text.", proba=None, text="")
    model = load_model()
    pred = model.predict([text])[0]
    # Some classifiers don't support predict_proba; our pipeline does.
    try:
        proba = model.predict_proba([text])[0]
        # probability of the predicted class
        pred_idx = list(model.classes_).index(pred)
        confidence = float(proba[pred_idx])
    except Exception:
        confidence = None
    return render_template("index.html", result=pred, proba=confidence, text=text)

if __name__ == "__main__":
    # Helpful default host for local dev
    app.run(host="127.0.0.1", port=5000, debug=True)
