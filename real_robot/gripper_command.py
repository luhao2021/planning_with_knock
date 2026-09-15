"""Shared nominal jaw-command calculation; not a measured aperture calibration."""
from xarm_executor import UnsafePlan


def pick_aperture_mm(cfg,kind,closing_width_mm):
    overrides=cfg.get('pick_aperture_mm_by_class',{})
    aperture=float(overrides.get(kind,closing_width_mm-cfg['gentle_compression_mm']))
    if not 0 < aperture <= min(closing_width_mm,cfg['gripper_open_mm']):
        raise UnsafePlan('Pick aperture command must be positive and no wider than the target or open gripper')
    return aperture
