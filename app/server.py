from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# paths
BASE_DIR = Path(__file__).resolve().parent          # .../app
REPO_DIR = BASE_DIR.parent                         # repo root
MODEL_PATH = REPO_DIR / "artifacts" / "models" / "emotion_classifier_tf.keras"
LABELMAP_PATH = REPO_DIR / "data" / "processed" / "label_mapping.json"

# mapping
with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
    emotion_to_id = json.load(f)
id_to_emotion = {v: k for k, v in emotion_to_id.items()}

# load model once (on startup)
model = tf.keras.models.load_model(MODEL_PATH)

# Extract the text vectorization layer from the model
# This assumes your model includes the TextVectorization layer as the first layer
vectorizer = model.layers[0]

app = FastAPI(title="Emotion Recognition Demo")

# serve static files (CSS/JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# serve the homepage
@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

class PredictRequest(BaseModel):
    text: str

@app.post("/api/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    # Convert text to lowercase (if your training did this)
    text = text.lower()
    
    # Vectorize the text using the model's vectorization layer
    # Create a batch of 1 text input
    text_input = tf.constant([text])
    
    # Get predictions
    probs = model.predict(text_input, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    pred_label = id_to_emotion[pred_id]
    confidence = float(np.max(probs))

    ranked = sorted(
        [{"label": id_to_emotion[i], "prob": float(p)} for i, p in enumerate(probs)],
        key=lambda x: x["prob"],
        reverse=True
    )

    return {"prediction": pred_label, "confidence": confidence, "ranked": ranked}