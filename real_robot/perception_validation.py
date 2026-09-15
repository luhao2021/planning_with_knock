"""Validate learned poses against their input masks/depth; never adjust a pose."""
from pathlib import Path
import itertools
import json
import numpy as np
from pose_to_plan import DIMENSIONS


class PoseValidationError(ValueError):
    """Registration exists but its fit is poor; eligible for one model retry."""
    def __init__(self,checks):
        self.checks=checks
        self.failed_names=[c['name'] for c in checks if not c['passed']]
        super().__init__('learned pose/mask/depth validation failed: '+', '.join(self.failed_names))


def validate_model_poses(records, capture_path, masks_path, output_path):
    import cv2
    with np.load(capture_path) as data:
        depth=data['depth']*float(data['depth_scale']); fx,fy,cx,cy=data['K']
    checks=[]
    with np.load(masks_path) as masks:
        for record in records:
            name=f'{record["class"]}_{record["instance_id"]}'
            if name not in masks: raise ValueError(f'{name}: learned pose has no matching instance mask')
            mask=masks[name].astype(bool); valid=mask & (depth>0)
            ys,xs=np.where(valid)
            if len(xs)<100: raise ValueError(f'{name}: insufficient valid masked depth')
            z=depth[ys,xs]
            camera=np.column_stack(((xs-cx)*z/fx,(ys-cy)*z/fy,z))
            t=np.asarray(record['T_camera_object'],dtype=float)
            local=(camera-t[:3,3])@t[:3,:3]
            half=np.asarray(DIMENSIONS[record['class']])/2
            q=np.abs(local)-half
            outside=np.linalg.norm(np.maximum(q,0),axis=1)
            inside=np.minimum(np.max(q,axis=1),0)
            residual=np.abs(outside+inside)
            support=float(np.mean(residual <= .012))
            median=float(np.median(residual)*1000)
            corners=np.asarray(list(itertools.product((-1,1),repeat=3)))*half
            projected=corners@t[:3,:3].T+t[:3,3]
            if np.any(projected[:,2]<=0): raise ValueError(f'{name}: model behind camera')
            pixels=np.column_stack((fx*projected[:,0]/projected[:,2]+cx,fy*projected[:,1]/projected[:,2]+cy))
            if not np.isfinite(pixels).all() or np.max(np.abs(pixels))>100000:
                raise ValueError(f'{name}: invalid model projection')
            silhouette=np.zeros(mask.shape,np.uint8)
            cv2.fillConvexPoly(silhouette,cv2.convexHull(np.round(pixels).astype(np.int32)),1)
            intersection=np.count_nonzero(mask & silhouette.astype(bool))
            coverage=intersection/max(int(mask.sum()),1)
            iou=intersection/max(np.count_nonzero(mask | silhouette.astype(bool)),1)
            passed=bool(median<=8 and support>=.60 and coverage>=.65 and iou>=.30)
            checks.append(dict(name=name,median_surface_residual_mm=median,depth_support_fraction=support,
                               mask_coverage=coverage,silhouette_iou=iou,passed=passed))
    Path(output_path).write_text(json.dumps(checks,indent=2)+'\n')
    failed=[c['name'] for c in checks if not c['passed']]
    if failed: raise PoseValidationError(checks)
    return checks
