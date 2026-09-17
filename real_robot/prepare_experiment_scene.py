#!/usr/bin/env python3
"""Prepare manual SAM2 seeds for an experiment; never connects to the robot."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_points(points, names, width, height):
    if set(points) != set(names):
        raise ValueError('Seed names must exactly match the configured object inventory')
    result = {}
    for name in names:
        xy = np.asarray(points[name], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f'{name}: expected a finite [pixel_x, pixel_y]')
        if not (0 <= xy[0] < width and 0 <= xy[1] < height):
            raise ValueError(f'{name}: point outside the capture')
        result[name] = [float(xy[0]), float(xy[1])]
    return result


def select_points(color, names, output_path):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(color[:, :, ::-1])  # RealSense capture stores BGR.
    points = {}
    for name in names:
        ax.set_title(f'Click INSIDE {name} on its top face\n'
                     'Left click: select; right click: undo; Enter: accept. Close: cancel.')
        fig.canvas.draw_idle()
        selected = plt.ginput(n=-1, timeout=0, show_clicks=True)
        if not plt.fignum_exists(fig.number) or len(selected) != 1:
            plt.close(fig)
            raise RuntimeError('Setup cancelled: select exactly one point, then press Enter for each box')
        points[name] = list(selected[0])
        ax.plot(*selected[0], '+', color='yellow', markersize=12)
        ax.annotate(name, selected[0], color='yellow', xytext=(5, 5), textcoords='offset points')
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='manual scene template')
    parser.add_argument('--capture', type=Path, help='existing aligned RGB-D NPZ; otherwise capture now')
    parser.add_argument('--points-json', type=Path, help='existing seed file for this exact capture, skipping GUI')
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text())
    perception = config['perception']
    if perception.get('automatic_scene', {}).get('enabled', False):
        parser.error('This template uses automatic Protex seeds; pass it directly to run_pipeline.py')
    names = list(perception['objects'])
    if not 1 <= len(names) <= 6:
        parser.error('Expected one to six configured boxes')
    for name, spec in perception['objects'].items():
        if name != f"{spec['class']}_{spec['instance_id']}" or spec['class'] not in ('protex', 'beach'):
            parser.error(f'Invalid experiment inventory entry: {name}')
        if not resolve(spec['mesh']).is_file():
            parser.error(f"Missing mesh: {spec['mesh']}; generate with make_box_mesh.py")
    folder = ROOT / 'real_robot/experiments/scene_setup' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
    folder.mkdir(parents=True)
    capture_path = folder / 'capture.npz'
    if args.capture:
        import shutil
        shutil.copyfile(resolve(args.capture), capture_path)
    else:
        from capture_rgbd import capture
        print('Capturing one RGB-D frame for scene setup; no robot connection.', flush=True)
        capture(capture_path, reset_before=True)
    with np.load(capture_path) as frame:
        color = frame['color']
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError('Expected a three-channel BGR capture')
    points = (json.loads(resolve(args.points_json).read_text()) if args.points_json else
              select_points(color, names, folder / 'selected_seeds.png'))
    points = validate_points(points, names, color.shape[1], color.shape[0])
    points_path = folder / 'points.json'
    points_path.write_text(json.dumps(points, indent=2) + '\n')
    perception['points'] = str(points_path.relative_to(ROOT))
    output_config = folder / 'pipeline_config.json'
    output_config.write_text(json.dumps(config, indent=2) + '\n')
    (folder / 'setup_metadata.json').write_text(json.dumps(dict(template=str(resolve(args.config)),
        capture=str(capture_path), instances=names, physical_motion=False), indent=2) + '\n')
    print('Prepared configuration:', output_config)
    print('Keep the boxes and camera in the same positions until the pipeline starts.')
    print(f'Preview: .venv-real/bin/python real_robot/run_pipeline.py --config {output_config}')
    print(f'Execute: .venv-real/bin/python real_robot/run_pipeline.py --config {output_config} --execute')


if __name__ == '__main__':
    main()
