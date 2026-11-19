"""Validate joins, frozen scores, bootstrap intervals, examples and checksums."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT, read_csv
from extract_image_features import extract
from reproduce_statistics import recompute


def verify(root=ROOT, output=None):
    root = Path(root)
    frames = read_csv(root / "data/per_frame.csv")
    conditions = read_csv(root / "data/per_condition.csv")
    assert len(frames) == 3027 and len(conditions) == 32
    keys = {(r["condition_id"], r["frame_id"]) for r in frames}
    assert len(keys) == 3027
    assert sum(int(r["n_frames"]) for r in conditions) == 3027
    assert {r["state_label"] for r in frames} == {f"S{k}" for k in range(6)}
    assert json.loads((root / "metadata/dataset.json").read_text())["state_labels"] == "All 3027 frames are manually annotated."
    for r in frames:
        assert int(r["state_score"]) == int(r["state_label"][1])
        area = int(r["area_left_px"]) + int(r["area_right_px"])
        span = int(r["span_left_px"]) + int(r["span_right_px"])
        assert area == int(r["exterior_area_px"])
        assert max(int(r["span_left_px"]), int(r["span_right_px"])) == int(r["ell_px"])
        assert int(r["mask_active"]) == int(area > 0)
        if area:
            assert abs(float(r["Ba"]) - int(r["area_right_px"]) / area) < 1e-14
            assert span and abs(float(r["Bl"]) - int(r["span_right_px"]) / span) < 1e-14
        else:
            assert r["Ba"] == r["Bl"] == ""
        assert abs(float(r["a"]) - area / (int(r["roi_width_px"])*int(r["roi_height_px"]))) < 6e-12
    rebuilt, report = recompute(root)
    archived = {r["condition_id"]: r for r in conditions}
    max_error = {}
    for row in rebuilt:
        original = archived[row["condition_id"]]
        for key, value in row.items():
            if key == "condition_id":
                continue
            if original[key] == "" or value == "":
                assert original[key] == value, (row["condition_id"], key)
                continue
            error = abs(float(value) - float(original[key]))
            max_error[key] = max(max_error.get(key, 0.), error)
            assert error <= 1e-10, (row["condition_id"], key, value, original[key])
    assert report["max_abs_PC1_error"] < 1e-12
    assert report["max_abs_refit_loading_error"] < 1e-12
    assert report["max_abs_SVD_loading_error"] < 1e-12
    processing = json.loads((root / "metadata/image_processing.json").read_text())
    frame_lookup = {(r["condition_id"], r["frame_id"]): r for r in frames}
    checks = []
    for example in read_csv(root / "examples/representatives.csv"):
        path = root / example["raw_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == example["image_sha256"]
        row = frame_lookup[(example["condition_id"], example["frame_id"])]
        assert row["state_label"] == example["state_label"]
        metrics, arrays = extract(path, processing)
        exact = ("ns", "nc", "area_left_px", "area_right_px", "span_left_px", "span_right_px", "ell_px",
                 "exterior_area_px", "estimated_jet_center_px", "estimated_base_width_px", "roi_width_px", "roi_height_px", "mask_active")
        for key in exact:
            assert int(metrics[key]) == int(row[key]), (example["state_label"], key)
        for key in ("a", "ell_um", "threshold_contrast", "valid_jet_row_fraction", "Ba", "Bl"):
            if row[key] == "":
                assert metrics[key] == ""
            else:
                assert abs(float(metrics[key]) - float(row[key])) < (1e-6 if key == "ell_um" else 1e-8), (example["state_label"], key)
        with Image.open(root / "examples/masks" / f"{example['state_label']}_exterior_mask.png") as image:
            assert np.array_equal(np.asarray(image) > 0, arrays["exterior"])
        checks.append({"state_label":example["state_label"], "raw_hash_verified":True, "mask_and_features_reproduced":True})
    checksums = root / "CHECKSUMS.sha256"
    verified_files = 0
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = (root / name).resolve()
            assert path.is_relative_to(root.resolve()), name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            verified_files += 1
    report.update(all_frame_relations_checked=True, all_32_condition_statistics_reproduced=True,
                  maximum_condition_errors=max_error, examples=checks,
                  checksum_files_verified=verified_files, raw_data_scope="Only six original frames are bundled; no full-3027-frame re-extraction is claimed.")
    if output is not None:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/validation_summary.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output), indent=2))
