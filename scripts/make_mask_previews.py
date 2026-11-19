"""Create clearly labeled derived mask previews; never edit a raw JPEG."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT, read_csv
from extract_image_features import extract


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mask_dir = root / "examples/masks"
    mask_dir.mkdir(exist_ok=True)
    parameters = json.loads((root / "metadata/image_processing.json").read_text())
    records = []
    for r in read_csv(root / "examples/representatives.csv"):
        metrics, arrays = extract(root / r["raw_path"], parameters)
        label = r["state_label"]
        roi = np.repeat(arrays["roi"][:, :, None], 3, axis=2)
        overlay = roi.astype(float)
        overlay[arrays["exterior"]] = .4 * overlay[arrays["exterior"]] + .6 * np.array([208,48,72])
        overlay = overlay.astype(np.uint8)
        cr = arrays["center_roi_px"]
        overlay[:, cr] = [0,110,110]
        dark = np.repeat((arrays["dark"].astype(np.uint8)*255)[:, :, None], 3, axis=2)
        images = [Image.fromarray(a) for a in (roi,dark,overlay)]
        w, h = images[0].size
        sheet = Image.new("RGB", (3*w + 24, h + 74), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (image, title) in enumerate(zip(images, ("Raw ROI (grayscale)","Threshold mask","Exterior mask overlay"))):
            x = 6 + i*(w+6)
            sheet.paste(image, (x,44)); draw.text((x,25), title, fill="black")
        draw.text((6,5), f"{label}: derived preview; red=exterior; teal=estimated axis", fill="black")
        draw.text((6,h+51), "Raw JPEG unchanged. Preview is not a physical bubble-location or temperature measurement.", fill="black")
        sheet.save(out / f"{label}_mask_preview.png")
        Image.fromarray(arrays["exterior"].astype(np.uint8)*255).save(mask_dir / f"{label}_exterior_mask.png")
        records.append({"state_label":label,"condition_id":r["condition_id"],"frame_id":r["frame_id"],**metrics})
    from common import write_csv
    write_csv(out / "example_features.csv", records)
    print("Six derived previews and masks created; originals unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples/previews")
    args = parser.parse_args()
    main(args.root, args.output_dir)
