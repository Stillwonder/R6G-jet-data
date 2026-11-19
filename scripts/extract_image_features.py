"""Extract the archived mask features from a raw frame, without changing labels.

The six bundled examples can be checked without the full raw-image collection.
Pass --raw-root to a full collection containing condition_id/frame_id files to
rerun every frame. Supplied manual state labels remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import ROOT, read_csv, write_csv


def components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(mask.shape[0]):
        for x in np.flatnonzero(mask[y] & ~seen[y]):
            if seen[y, x]:
                continue
            queue = [(y, int(x))]
            seen[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if size >= 2:
                sizes.append(size)
    return sizes


def extract(path, parameters):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    top = gray[:min(260, len(gray))]
    darkness = (255. - top).mean(axis=0)
    lo = max(20, gray.shape[1] // 8)
    hi = min(gray.shape[1] - 20, gray.shape[1] * 7 // 8)
    center = int(lo + np.argmax(darkness[lo:hi]))
    base_mask = top < 105
    widths = []
    for row in base_mask:
        xs = np.flatnonzero(row[max(0, center - 90):min(gray.shape[1], center + 91)])
        if len(xs) >= 3:
            widths.append(int(xs[-1] - xs[0] + 1))
    base = int(np.clip(np.median(widths), 8, 80)) if widths else 24
    y_end = max(100, gray.shape[0] - 100)
    half = min(180, gray.shape[1] // 2 - 2)
    x0, x1 = max(0, center - half), min(gray.shape[1], center + half)
    roi = gray[:y_end, x0:x1]
    background = np.asarray(Image.fromarray(roi.astype(np.uint8)).filter(ImageFilter.GaussianBlur(18)), dtype=np.float32)
    contrast = background - roi
    threshold = float(np.clip(np.percentile(contrast, 97), 30, 65))
    dark = contrast > threshold
    cr = int(np.clip(center - x0, 0, roi.shape[1] - 1))
    valid = np.zeros(len(roi), dtype=bool)
    left = np.zeros(len(roi), dtype=int)
    right = np.zeros(len(roi), dtype=int)
    for y, row in enumerate(dark):
        xs = np.flatnonzero(row)
        if not len(xs):
            continue
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        distance = lambda values: 0 if values[0] <= cr <= values[-1] else min(abs(values[0] - cr), abs(values[-1] - cr))
        run = min(runs, key=distance)
        if distance(run) <= max(20, 2 * base):
            valid[y] = True
            left[y] = max(0, cr - int(run[0]))
            right[y] = max(0, int(run[-1]) - cr)
    half_band = max(5, int(base * .85))
    exterior = dark.copy()
    exterior[:, max(0, cr - half_band):min(roi.shape[1], cr + half_band + 1)] = False
    exterior[~valid] = False
    active_rows = valid & (exterior.sum(axis=1) >= max(3, int(base * .25)))
    lengths = []
    run = 0
    for value in np.r_[active_rows, False]:
        if value:
            run += 1
        else:
            if run >= 2:
                lengths.append(run)
            run = 0
    nc = len(components(exterior[::4, ::4]))
    a_left, a_right = int(exterior[:, :cr].sum()), int(exterior[:, cr + 1:].sum())
    l_left, l_right = int(left.max()), int(right.max())
    active = a_left + a_right > 0
    metrics = {"a": float(exterior.sum() / exterior.size), "ell_um": max(l_left, l_right) * parameters["um_per_px"],
               "ns": len(lengths), "nc": nc, "area_left_px": a_left, "area_right_px": a_right,
               "span_left_px": l_left, "span_right_px": l_right, "ell_px": max(l_left, l_right),
               "exterior_area_px": int(exterior.sum()), "mask_active": int(active),
               "Ba": a_right / (a_left + a_right) if active else "",
               "Bl": l_right / (l_left + l_right) if active and l_left + l_right else "",
               "estimated_jet_center_px": center, "estimated_base_width_px": base,
               "roi_width_px": roi.shape[1], "roi_height_px": roi.shape[0],
               "threshold_contrast": threshold, "valid_jet_row_fraction": float(valid.mean())}
    return metrics, {"roi": roi.astype(np.uint8), "dark": dark, "exterior": exterior,
                     "center_roi_px": cr, "half_band_px": half_band, "roi_x0_px": x0}


def run(root, output, raw_root=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((root / "metadata/image_processing.json").read_text())
    records = []
    for row in read_csv(root / "data/per_frame.csv"):
        if raw_root is None:
            if not int(row["raw_image_included"]):
                continue
            path = root / row["example_path"]
        else:
            path = Path(raw_root) / row["condition_id"] / row["frame_id"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != row["image_sha256"]:
            raise ValueError(f"Original image hash mismatch: {row['condition_id']}/{row['frame_id']}")
        metrics, arrays = extract(path, parameters)
        records.append({"condition_id": row["condition_id"], "frame_id": row["frame_id"], **metrics})
        if raw_root is None:
            label = row["state_label"]
            Image.fromarray(arrays["exterior"].astype(np.uint8) * 255).save(output / f"{label}_exterior_mask.png")
    write_csv(output / "features_recomputed.csv", records)
    print(f"Extracted {len(records)} frames; state labels were not reassigned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/example_check")
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()
    run(args.root, args.output_dir, args.raw_root)
