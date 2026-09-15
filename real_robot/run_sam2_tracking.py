#!/usr/bin/env python3
"""Official SAM 2 video tracking over one image per completed robot action.

No intermediate robot-motion frames are captured. The original prompts initialise
stable IDs; the current observation is the last numbered JPEG in --frames.
"""
import argparse
import json
from pathlib import Path


def check_mask_identity(masks, required=None):
    """Ignore ambiguity between unrelated tracks; protect requested identities."""
    for i, a in enumerate(masks):
        for b in list(masks)[i+1:]:
            if required is not None and a not in required and b not in required:
                continue
            overlap = int((masks[a] & masks[b]).sum()) / min(int(masks[a].sum()), int(masks[b].sum()))
            if overlap > 0.10:
                raise RuntimeError(f'ambiguous tracked instances {a}, {b}: overlap {overlap:.1%}')


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from sam2.build_sam import build_sam2_video_predictor
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--required-instances',nargs='+',help='only these identities can fail the current observation')
    ap.add_argument('--active-instances',nargs='+',help='only initialize these remaining physical identities')
    ap.add_argument('--defer-identity-check',action='store_true',help='spatial association validates source masks in the parent process')
    args = ap.parse_args()
    prompts = json.loads(args.points.read_text())
    frames = sorted(args.frames.glob('*.jpg'))
    if not prompts or not frames:
        raise ValueError('initial prompts and at least one frame are required')
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(args.checkpoint.resolve()), device='cuda')
    names = sorted(prompts if args.active_instances is None else args.active_instances)
    if len(names)!=len(set(names)) or not set(names)<=set(prompts):
        raise ValueError('active names must be distinct initial identities')
    last = None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(args.frames.resolve()), offload_video_to_cpu=True)
        for index, name in enumerate(names, 1):
            coords = np.asarray([prompts[name]] + [prompts[n] for n in names if n != name], dtype=np.float32)
            predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=index,
                                           points=coords, labels=np.asarray([1]+[0]*(len(names)-1), dtype=np.int32))
        for frame_idx, ids, logits in predictor.propagate_in_video(state):
            if frame_idx == len(frames)-1:
                last = (ids, logits.detach().float().cpu().numpy())
    if last is None:
        raise RuntimeError('SAM 2 did not return the latest frame')
    args.out.mkdir(parents=True, exist_ok=True)
    image = Image.open(frames[-1]).convert('RGB'); draw = ImageDraw.Draw(image)
    masks = {}; records = []
    for index, obj_id in enumerate(last[0]):
        name = names[int(obj_id)-1]; mask = last[1][index].squeeze() > 0
        ys, xs = np.where(mask)
        record = {'instance_id': name, 'pixels': int(mask.sum()), 'tracked': bool(len(xs) >= 100)}
        if record['tracked']:
            box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
            record['bbox_xyxy'] = box; masks[name] = mask
            draw.rectangle(box, outline='red', width=2); draw.text(box[:2], name, fill='red')
        records.append(record)
    np.savez_compressed(args.out/'instance_masks.npz', **masks)
    (args.out/'instance_masks.json').write_text(json.dumps(records, indent=2)+'\n')
    image.save(args.out/'tracked_masks_preview.png')
    if not args.defer_identity_check:
        check_mask_identity(masks,None if args.required_instances is None else set(args.required_instances))
    print(json.dumps(records))


if __name__ == '__main__':
    main()
