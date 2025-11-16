import pickle
from typing import Literal
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path


# Sample Data
# sample_data = {
# "cap_diameter": 15.26,
# "stem_height": 16.95,
# "stem_width": 17.09,
# "gill_color": "w",
# "does_bruise_or_bleed": "f",
# "stem_surface": "y",
# "cap_shape": "x",
# "habitat": "d",
# "gill_attachment": "e",
# "season": "w",
# "ring_type": "g",
# "cap_surface": "g",
# "cap_color": "o",
# "has_ring": "t",
# "gill_spacing": "unknown",
# "stem_color": "w"
# }


app = FastAPI(title="Mushroom Classification")
model = pickle.load(open("models/model.pkl", "rb"))


class Mushroom(BaseModel):
    cap_diameter: float = Field(..., ge=0)
    stem_height: float = Field(..., ge=0)
    stem_width: float = Field(..., ge=0)
    gill_color: Literal["w", "n", "p", "u", "b", "g", "y", "r", "e", "o", "k", "f"]
    does_bruise_or_bleed: Literal["f", "t"]
    stem_surface: Literal["y", "unknown", "s", "k", "i", "h", "t", "g", "f"]
    cap_shape: Literal["x", "f", "p", "b", "c", "s", "o"]
    habitat: Literal["d", "m", "g", "h", "l", "p", "w", "u"]
    gill_attachment: Literal["e", "unknown", "a", "d", "s", "x", "p", "f"]
    season: Literal["w", "u", "a", "s"]
    ring_type: Literal["g", "p", "e", "l", "f", "m", "unknown", "r", "z"]
    cap_surface: Literal["g", "h", "unknown", "t", "y", "e", "s", "l", "d", "w", "i", "k"]
    cap_color: Literal["o", "e", "n", "g", "r", "w", "y", "p", "u", "b", "l", "k"]
    has_ring: Literal["t", "f"]
    gill_spacing: Literal["unknown", "c", "d", "f"]
    stem_color: Literal["w", "y", "n", "u", "b", "l", "r", "p", "e", "k", "g", "o", "f"]


class PredictResponse(BaseModel):
    pred: Literal["p", "e"]
    predict: Literal["poisonous", "edible"]

def predict_single(mushroom):
    df = pd.DataFrame([mushroom])
    return model.predict(df)[0]

@app.post("/predict")
def predict(mushroom: Mushroom) -> PredictResponse:
    result = predict_single(mushroom.model_dump())
    return PredictResponse(pred=result, predict="poisonous" if result == "p" else "edible")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)