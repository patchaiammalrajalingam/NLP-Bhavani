from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("toxic_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict_toxicity(data: TextInput):

    text_vector = vectorizer.transform([data.text])
    prob = model.predict_proba(text_vector)[0][1]

    result = "Toxic" if prob >= 0.5 else "Non-Toxic"

    return {
        "prediction": result,
        "toxicity_score": float(prob)
    }