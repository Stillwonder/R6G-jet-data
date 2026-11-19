"""Small shared helpers for the archived CSV/PCA data contract."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_pc1(frames, parameters):
    values = np.array([[float(row[key]) for key in parameters["feature_columns"]]
                       for row in frames], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("PCA inputs must be finite nonnegative archived features.")
    sd = np.asarray(parameters["log_sample_sd"], dtype=np.float64)
    if np.any(sd <= 0):
        raise ValueError("Standardization requires positive sample deviations.")
    standardized = (np.log1p(values) - parameters["log_mean"]) / sd
    return standardized @ np.asarray(parameters["loadings"], dtype=np.float64), standardized


def load_parameters(root=ROOT):
    return json.loads((Path(root) / "metadata/pca_parameters.json").read_text(encoding="utf-8"))
