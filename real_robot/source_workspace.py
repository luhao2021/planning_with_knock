"""Source-area bounds; tolerance is for tracking after initial scene setup."""
import numpy as np
from xarm_executor import UnsafePlan


def source_xy_bounds(workflow, *, with_tolerance=False):
    lo=np.asarray(workflow['source_xy_min_mm'],float)
    hi=np.asarray(workflow['source_xy_max_mm'],float)
    try:tolerance=float(workflow.get('source_boundary_tolerance_mm',0))
    except (TypeError,ValueError):raise UnsafePlan('source_boundary_tolerance_mm must be a number') from None
    if not np.isfinite(tolerance) or not 0<=tolerance<=100:
        raise UnsafePlan('source_boundary_tolerance_mm must be between 0 and 100 mm')
    if with_tolerance:return lo-tolerance,hi+tolerance
    return lo,hi
