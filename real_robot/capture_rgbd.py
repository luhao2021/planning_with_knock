#!/usr/bin/env python3
"""Save one aligned RealSense color/depth frame; no xArm connection."""
import argparse
import time
from pathlib import Path
import numpy as np
import pyrealsense2 as rs


def reset_camera():
    """Reset the connected RealSense and wait for that serial to reappear."""
    context = rs.context()
    devices = list(context.query_devices())
    if len(devices) != 1:
        raise RuntimeError(f'Expected one RealSense camera for startup reset; found {len(devices)}')
    device = devices[0]
    serial = device.get_info(rs.camera_info.serial_number)
    print(f'[CAMERA] Resetting RealSense {serial}; waiting for USB reconnection.', flush=True)
    device.hardware_reset()
    del device, devices, context
    # A reset disconnects the device. Give USB time to re-enumerate before opening it.
    time.sleep(5)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            context = rs.context()
            if any(d.get_info(rs.camera_info.serial_number) == serial for d in context.query_devices()):
                print(f'[CAMERA] RealSense {serial} reconnected.', flush=True)
                return serial
        except RuntimeError:
            pass  # USB enumeration may be temporarily unavailable during reset.
        time.sleep(.5)
    raise RuntimeError(f'RealSense {serial} did not reconnect after startup reset; check its USB connection')


def capture(path: Path, *, reset_before: bool = True):
    serial = reset_camera() if reset_before else None
    pipeline = rs.pipeline()
    config = rs.config()
    if serial is not None:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
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
