# Fake News Detector (ML + Flask)

A complete, working project to detect fake news headlines/articles using a machine-learning pipeline (TF-IDF + Logistic Regression) with a simple Flask web interface.

## Quick Start

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train (optional)**  
   By default, a small toy model is already trained and saved in `models/model.joblib` so you can run the app immediately.  
   To retrain on your own dataset (CSV with columns: `text`, `label` where label is `FAKE` or `REAL`), put it at `data/news.csv` and run:
   ```bash
   python train.py --data data/news.csv
   ```
   You can also let `train.py` fall back to the included toy dataset:
   ```bash
   python train.py
   ```

4. **Run the web app**
   ```bash
   python app.py
   ```
   Then open your browser at http://127.0.0.1:5000

## Files

- `app.py` — Flask server with a minimal UI for predictions.
- `train.py` — Model training script (TF-IDF + Logistic Regression). Saves to `models/model.joblib`.
- `models/model.joblib` — Pretrained model (toy dataset) so the app works out-of-the-box.
- `data/toy.csv` — Small, included dataset used for default training.
- `templates/index.html` — Webpage template.
- `static/style.css` — Basic styling.

## Notes
- For best accuracy, use a large, real dataset. You can adapt any fake-news dataset to the `text,label` schema.
- `train.py` prints basic metrics and writes them to `models/metrics.json`.
