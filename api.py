
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

class IncomeMLP(nn.Module):

    def __init__(self, input_size, hidden_sizes,
                 dropout=0.0, batch_norm=False):
        super().__init__()

        layers = []
        previous = input_size

        for width in hidden_sizes:
            layers.append(nn.Linear(previous, width))

            if batch_norm:
                layers.append(nn.BatchNorm1d(width))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous = width

        layers.append(nn.Linear(previous, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# STEP 2: Keep configurations identical to the ones in train_models.py.
MODEL_CONFIGS = {
    "Model_1_Simple_MLP": {"hidden_sizes": [32], "dropout": 0.0, "batch_norm": False},
    "Model_2_Deep_MLP": {"hidden_sizes": [128, 64, 32], "dropout": 0.0, "batch_norm": False},
    "Model_3_Regularised_MLP": {"hidden_sizes": [128, 64, 32], "dropout": 0.25, "batch_norm": True},
}

# STEP 3: Load the fitted preprocessor and the best model once at startup.
BASE_DIR = Path(__file__).resolve().parent

OUTPUTS = BASE_DIR / "outputs"

preprocessor_path = OUTPUTS / "preprocessor.joblib"
champion_info_path = OUTPUTS / "champion_info.json"

if not preprocessor_path.is_file() or not champion_info_path.is_file():
    raise FileNotFoundError(
        "Missing outputs/preprocessor.joblib or outputs/champion_info.json. "
        "Run the training project successfully first."
    )

# Only load trusted .joblib artifacts created by your own training.
preprocessor = joblib.load(preprocessor_path)
with champion_info_path.open(encoding="utf-8") as info_file:
    champion_info = json.load(info_file)
if not isinstance(champion_info, dict):
    raise ValueError("champion_info.json must contain a JSON object.")
champion_name = champion_info.get("champion_name")
if not isinstance(champion_name, str) or not champion_name:
    raise ValueError("champion_info.json must contain a non-empty champion_name.")
if champion_name not in MODEL_CONFIGS:
    raise ValueError(f"Unknown model architecture: {champion_name}")

weights_path = OUTPUTS / f"{champion_name}.pt"
if not weights_path.is_file():
    raise FileNotFoundError(f"Champion model weights missing: {weights_path}")

weights = torch.load(weights_path, map_location="cpu", weights_only=True)
input_size = weights["network.0.weight"].shape[1]
model = IncomeMLP(input_size=input_size, **MODEL_CONFIGS[champion_name])
model.load_state_dict(weights)
model.eval()  # Required for inference with Dropout/BatchNorm

app = FastAPI(title="Adult Census Income Prediction API", version="1.0.0")


# STEP 4: Define the request body using the 12 features used in training.
# The aliases preserve original Adult CSV column names with dots.
class PersonInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: int = Field(ge=0, le=120, examples=[39])
    workclass: str | None = Field(default=None, examples=["State-gov"])
    education_num: int = Field(alias="education.num", ge=1, le=16, examples=[13])
    marital_status: str | None = Field(
        default=None, alias="marital.status", examples=["Never-married"]
    )
    occupation: str | None = Field(default=None, examples=["Adm-clerical"])
    relationship: str | None = Field(default=None, examples=["Not-in-family"])
    race: str | None = Field(default=None, examples=["White"])
    sex: str | None = Field(default=None, examples=["Male"])
    capital_gain: int = Field(alias="capital.gain", ge=0, examples=[2174])
    capital_loss: int = Field(alias="capital.loss", ge=0, examples=[0])
    hours_per_week: int = Field(alias="hours.per.week", ge=1, le=168, examples=[40])
    native_country: str | None = Field(
        default=None, alias="native.country", examples=["United-States"]
    )


@app.get("/")
def home():
    return {"message": "Adult income prediction API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "champion_model": champion_name}


# STEP 5: Endpoint: JSON -> dataframe -> saved preprocessing -> model -> JSON.
@app.post("/predict")
def predict(person: PersonInput):
    row = person.model_dump(by_alias=True)
    for key, value in row.items():
        if isinstance(value, str):
            row[key] = value.strip()

    data = pd.DataFrame([row]).replace("?", np.nan)
    # Reproduce the exact feature order used to fit the preprocessor.
    data = data.loc[:, list(preprocessor.feature_names_in_)]
    processed = preprocessor.transform(data).astype(np.float32)
    inputs = torch.from_numpy(processed)

    with torch.inference_mode():
        logits = model(inputs)
        probability = float(torch.sigmoid(logits).item())

    predicted_class = int(probability >= 0.5)
    return {
        "champion_model": champion_name,
        "predicted_class": predicted_class,
        "prediction": ">50K" if predicted_class == 1 else "<=50K",
        "probability_over_50k": round(probability, 4),
    }