"""Load the saved champion model and generate income predictions from a CSV."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


class IncomeMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.0, batch_norm=False):
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


def predict_csv(input_csv, output_csv=None, artifacts_dir="outputs"):
    """Predict income for rows in a CSV using the saved champion artifacts."""
    artifacts_dir = Path(artifacts_dir)
    metadata_path = artifacts_dir / "champion_info.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. Run the notebook training cells first."
        )

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    preprocessor_path = artifacts_dir / metadata["preprocessor_file"]
    weights_path = artifacts_dir / metadata["champion_weights"]
    for artifact_path in (preprocessor_path, weights_path):
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Required model artifact not found: {artifact_path}. "
                "Run the notebook training cells first."
            )

    rows = pd.read_csv(input_csv)
    feature_columns = metadata["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in rows.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required model features: "
            + ", ".join(missing_columns)
        )

    preprocessor = joblib.load(preprocessor_path)
    features = preprocessor.transform(rows[feature_columns]).astype(np.float32)
    if features.shape[1] != metadata["input_size"]:
        raise ValueError(
            f"Preprocessor produced {features.shape[1]} features, "
            f"but the model expects {metadata['input_size']}."
        )

    model = IncomeMLP(
        metadata["input_size"],
        metadata["hidden"],
        metadata["dropout"],
        metadata["batch_norm"],
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            batch = torch.from_numpy(features[start : start + 2048])
            logits = model(batch)
            probabilities.extend(torch.sigmoid(logits).numpy().ravel())

    probabilities = np.asarray(probabilities, dtype=np.float64)
    threshold = metadata["threshold"]
    predicted_classes = (probabilities >= threshold).astype(int)
    rows["probability_above_50k"] = probabilities
    rows["predicted_class"] = predicted_classes
    rows["predicted_income"] = np.where(predicted_classes == 1, ">50K", "<=50K")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output_path, index=False)

    return rows
