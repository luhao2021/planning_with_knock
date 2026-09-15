#!/usr/bin/env python3
"""Run official SAM 2.1 on one supplied positive pixel per instance.

The points are image prompts, not pose estimates. FoundationPose consumes the
resulting masks and produces 6D poses once its runtime is installed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--points', type=Path, required=True,
                    help='JSON mapping stable instance IDs to [u,v] prompt pixels')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    prompts = json.loads(args.points.read_text())
    with np.load(args.capture) as data:
        rgb = data['color'][..., ::-1].copy()
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                       str(args.checkpoint.resolve()), device='cuda')
    predictor = SAM2ImagePredictor(model)
    args.out.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    masks_out = {}
    records = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.set_image(rgb)
        for name, positive in prompts.items():
            coords = np.array([positive] + [p for other, p in prompts.items() if other != name], np.float32)
            labels = np.array([1] + [0] * (len(coords)-1), np.int32)
            masks, scores, _ = predictor.predict(point_coords=coords,
                                                  point_labels=labels,
                                                  multimask_output=True)
            idx = int(np.argmax(scores))
            mask = masks[idx].astype(bool)
            if not mask[round(positive[1]), round(positive[0])]:
                raise ValueError(f'{name}: prompt absent from predicted mask')
            ys, xs = np.where(mask)
            if len(xs) < 100:
                raise ValueError(f'{name}: mask too small')
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
            masks_out[name] = mask
            records.append(dict(instance_id=name, point=positive, bbox_xyxy=bbox,
                                pixels=len(xs), sam_score=float(scores[idx])))
            draw.rectangle(bbox, outline='red', width=2)
            draw.text((bbox[0], bbox[1]), name, fill='red')
    np.savez_compressed(args.out/'instance_masks.npz', **masks_out)
    image.save(args.out/'prompted_masks_preview.png')
    (args.out/'instance_masks.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
