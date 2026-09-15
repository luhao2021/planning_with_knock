#!/usr/bin/env python3
"""Run official SAM 2.1 automatic instance masks on an RGB-D capture.

This is learned segmentation only; FoundationPose must still register each
accepted mask before the solver receives any 6D pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--point-grid',type=Path,help='NPZ containing normalized full-image point_grid for depth-directed automatic prompts')
    ap.add_argument('--roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                    default=[230, 210, 430, 370])
    ap.add_argument('--refine-proposals',action='store_true',help='bounded initialization retry: retain less certain proposals for geometric filtering')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for SAM 2 inference')
    with np.load(args.capture) as data:
        bgr = data['color']
    # RealSense capture is BGR; SAM expects RGB.
    rgb = bgr[..., ::-1].copy()
    x0, y0, x1, y1 = (0,0,rgb.shape[1],rgb.shape[0]) if args.point_grid else args.roi
    crop = rgb[y0:y1, x0:x1]
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                       str(args.checkpoint.resolve()), device='cuda')
    options=dict(points_per_side=16,points_per_batch=32,min_mask_region_area=30)
    if args.point_grid:
        with np.load(args.point_grid) as data:grid=data['point_grid']
        if grid.ndim!=2 or grid.shape[1]!=2 or not len(grid) or not np.isfinite(grid).all() or np.any((grid<0)|(grid>=1)):
            raise ValueError('automatic point grid must contain normalized image coordinates')
        options.update(points_per_side=None,point_grids=[grid],pred_iou_thresh=.7,stability_score_thresh=.9)
    if args.refine_proposals:options.update(pred_iou_thresh=.6,stability_score_thresh=.85)
    generator = SAM2AutomaticMaskGenerator(model,**options)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        candidates = generator.generate(crop)
    args.out.mkdir(parents=True, exist_ok=True)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    records = []
    masks = {}
    for i, item in enumerate(sorted(candidates, key=lambda m: m['predicted_iou'], reverse=True)):
        mask = np.asarray(item['segmentation'], dtype=bool)
        area = int(mask.sum())
        if area < 150 or area > 16000:
            continue
        xx, yy, ww, hh = map(int, item['bbox'])
        name = f'candidate_{len(records)+1:02d}'
        full = np.zeros(rgb.shape[:2], dtype=bool)
        full[y0:y1, x0:x1] = mask
        masks[name] = full
        records.append(dict(name=name, bbox_xywh=[xx+x0, yy+y0, ww, hh],
                            area_px=area, predicted_iou=float(item['predicted_iou']),
                            stability_score=float(item['stability_score'])))
        draw.rectangle((xx+x0, yy+y0, xx+x0+ww, yy+y0+hh), outline='red', width=2)
        draw.text((xx+x0, yy+y0), name, fill='red')
    canvas.save(args.out / 'sam2_candidates.png')
    np.savez_compressed(args.out / 'sam2_masks.npz', **masks)
    (args.out / 'sam2_candidates.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
