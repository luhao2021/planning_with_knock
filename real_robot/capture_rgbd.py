#!/usr/bin/env python3
"""Save one aligned RealSense color/depth frame; no xArm connection."""
import argparse
from pathlib import Path
import numpy as np
import pyrealsense2 as rs


def capture(path: Path):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(45):
            frames = align.process(pipeline.wait_for_frames(5000))
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError('aligned RGB-D frame missing')
        intr = color.profile.as_video_stream_profile().intrinsics
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, color=np.asarray(color.get_data()),
                            depth=np.asarray(depth.get_data()),
                            K=np.array([intr.fx,intr.fy,intr.ppx,intr.ppy]),
                            depth_scale=profile.get_device().first_depth_sensor().get_depth_scale(),
                            serial=profile.get_device().get_info(rs.camera_info.serial_number))
    finally:
        pipeline.stop()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    capture(args.out)
    print(args.out)
