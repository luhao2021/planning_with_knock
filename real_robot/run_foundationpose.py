#!/usr/bin/env python3
"""Register SAM 2 instance masks with NVIDIA's pretrained FoundationPose.

Requires the official compiled libfoundation_pose_nvidia.so, official Python
bindings, and ONNX weights. This process never connects to the xArm.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from foundation_pose_nvidia import MultiObjectEstimator, ObjectSpec, RegisterOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--refiner', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--object-specs', type=Path, help='JSON mapping mask names to class, instance_id and mesh')
    parser.add_argument('--engine-cache', type=Path)
    parser.add_argument('--n-hypotheses', type=int, default=64)
    parser.add_argument('--n-refine', type=int, default=3)
    parser.add_argument('--previous-capture',type=Path,help='recovery only: register previous frame then track current frame')
    parser.add_argument('--previous-masks',type=Path)
    parser.add_argument('--instances',nargs='+',help='process only these named instances')
    args = parser.parse_args()
    if bool(args.previous_capture)!=bool(args.previous_masks):
        parser.error('--previous-capture and --previous-masks must be supplied together')
    for path in (args.capture, args.masks, args.mesh, args.library, args.refiner, args.scorer):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.capture) as data:
        rgb = np.ascontiguousarray(data['color'][..., ::-1], np.uint8)
        depth = np.ascontiguousarray(data['depth'] * float(data['depth_scale']), np.float32)
        fx, fy, cx, cy = data['K']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
    with np.load(args.masks) as data:
        names = sorted(data.files)
        masks = {name: np.ascontiguousarray(data[name], np.uint8) for name in names}
    if args.instances:
        missing=set(args.instances)-set(names)
        if missing:raise ValueError(f'requested instances missing current masks: {sorted(missing)}')
        names=sorted(set(args.instances));masks={name:masks[name] for name in names}
    if not names:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'foundationpose_instances.json').write_text(json.dumps({'instances': []}))
        return
    if any(mask.sum() < 100 for mask in masks.values()):
        raise ValueError('nonempty, distinct instance masks required')
    metadata = json.loads(args.object_specs.read_text()) if args.object_specs else {}
    specs = [ObjectSpec(name=name, cad_path=Path(metadata.get(name, {}).get('mesh', args.mesh)).resolve(), mesh_unit_scale=1.0)
             for name in names]
    options = RegisterOptions(library_path=args.library.resolve(),
                              refine_model_path=args.refiner.resolve(),
                              score_model_path=args.scorer.resolve(),
                              engine_cache_dir=(args.engine_cache or args.out/'engine_cache').resolve(),
                              score_output_name='score_logit',
                              n_hypotheses=args.n_hypotheses, n_refine=args.n_refine)
    frame = SimpleNamespace(rgb=rgb, k=K)
    with MultiObjectEstimator(options) as estimator:
        if args.previous_capture:
            with np.load(args.previous_capture) as data:
                previous_rgb=np.ascontiguousarray(data['color'][...,::-1],np.uint8)
                previous_depth=np.ascontiguousarray(data['depth']*float(data['depth_scale']),np.float32)
                pfx,pfy,pcx,pcy=data['K']
                previous_k=np.array([[pfx,0,pcx],[0,pfy,pcy],[0,0,1]],np.float32)
            with np.load(args.previous_masks) as data:
                previous_masks={name:np.ascontiguousarray(data[name],np.uint8) for name in names}
            previous_frame=SimpleNamespace(rgb=previous_rgb,k=previous_k)
            estimator.register(previous_frame,specs,previous_masks,previous_depth)
            predictions=estimator.track(frame,depth,n_refine=args.n_refine)
        else:
            predictions = estimator.register(frame, specs, masks, depth)
    records = []
    for pred in predictions:
        if pred.status != 0:
            raise RuntimeError(f'{pred.object_spec.name}: FoundationPose {pred.status_name}')
        meta = metadata.get(pred.object_spec.name, {})
        records.append({'class': meta.get('class', 'protex'), 'instance_id': meta.get('instance_id', pred.object_spec.name.split('_')[-1]),
                        'score': float(pred.score),
                        'T_camera_object': pred.pose.astype(float).tolist()})
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'foundationpose_instances.json'
    path.write_text(json.dumps({'instances': records}, indent=2))
    print(path)


if __name__ == '__main__':
    main()
