"""Recompute PC1 and condition bootstrap intervals from the archived frame table.

Manually annotated state labels are inputs to this script. Stable
condition/row order and the S,PC1,Ba,Bl draw order reproduce manuscript CIs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, load_parameters, project_pc1, read_csv, write_csv


def bootstrap(values, rng, resamples=2000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return "", "", ""
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975], method="linear")
    return float(values.mean()), float(lo), float(hi)


def recompute(root=ROOT, out=None, seed=20261115, resamples=2000):
    root = Path(root)
    frames = sorted(read_csv(root / "data/per_frame.csv"), key=lambda r: int(r["row_index"]))
    conditions = sorted(read_csv(root / "data/per_condition.csv"), key=lambda r: int(r["condition_order"]))
    score, z = project_pc1(frames, load_parameters(root))
    stored = np.array([float(r["PC1"]) for r in frames])
    rng = np.random.default_rng(seed)
    rebuilt = []
    for condition in conditions:
        indices = [i for i, r in enumerate(frames) if r["condition_id"] == condition["condition_id"]]
        active = [i for i in indices if int(frames[i]["mask_active"])]
        row = {key: condition[key] for key in ("condition_order", "condition_id", "concentration_molL",
               "energy_mJ", "laser_voltage_V", "delay_us")}
        row.update(n_frames=len(indices), n_active=len(active),
                   n_nonS0=sum(int(frames[i]["state_score"]) != 0 for i in indices))
        metrics = (("S_bar", [int(frames[i]["state_score"]) for i in indices], "S"),
                   ("PC1_mean", score[indices], "PC1"),
                   ("Ba_mean", [float(frames[i]["Ba"]) for i in active], "Ba"),
                   ("Bl_mean", [float(frames[i]["Bl"]) for i in active], "Bl"))
        for name, values, prefix in metrics:
            mean, lo, hi = bootstrap(values, rng, resamples)
            row.update({name: mean, prefix + "_ci95_lower": lo, prefix + "_ci95_upper": hi})
        for k in range(6):
            count = sum(int(frames[i]["state_score"]) == k for i in indices)
            row[f"S{k}_count"] = count
            row[f"S{k}_fraction"] = count / len(indices)
        for key in ("a", "ell_um", "ns", "nc"):
            values = np.array([float(frames[i][key]) for i in indices])
            row[key + "_mean"] = float(values.mean())
            row[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
        rebuilt.append(row)
    covariance = np.cov(z, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    v = eigenvectors[:, -1]
    if v.sum() < 0:
        v = -v
    parameters = load_parameters(root)
    reference = np.asarray(parameters["loadings"])
    _, singular_values, right_vectors = np.linalg.svd(z, full_matrices=False)
    svd_loading = right_vectors[0]
    if svd_loading.sum() < 0:
        svd_loading = -svd_loading
    checks = {"frames": len(frames), "conditions": len(conditions), "seed": seed,
              "resamples": resamples, "rng": "NumPy default_rng / PCG64",
              "max_abs_PC1_error": float(np.max(np.abs(score - stored))),
              "max_abs_refit_loading_error": float(np.max(np.abs(v - reference))),
              "max_abs_SVD_loading_error": float(np.max(np.abs(svd_loading - reference))),
              "explained_variance_fraction_refit": float(eigenvalues[-1] / eigenvalues.sum()),
              "explained_variance_fraction_SVD": float(singular_values[0]**2 / np.sum(singular_values**2)),
              "eigenpair_residual_l2": float(np.linalg.norm(covariance @ reference - eigenvalues[-1] * reference)),
              "draw_order": "condition_order; within each: S, PC1, Ba, Bl"}
    if out is not None:
        out = Path(out)
        if out.resolve() == (root / "data").resolve():
            raise ValueError("Output must not overwrite the archived data directory.")
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "per_condition_recomputed.csv", rebuilt)
        write_csv(out / "pc1_recomputed.csv", [{"row_index": r["row_index"], "condition_id": r["condition_id"],
                   "frame_id": r["frame_id"], "PC1_recomputed": float(p)} for r, p in zip(frames, score)])
        (out / "statistics_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return rebuilt, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/recomputed")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    _, report = recompute(args.root, args.output_dir, args.seed, args.resamples)
    print(json.dumps(report, indent=2))
