"""Associate source masks with fixed physical identities; no robot access.

SAM labels are proposals. Names belong to the initial registry, never to bowls.
Only a Knock target gets a motion corridor; other objects use their last pose.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from pose_to_plan import base_to_optical


def base_depth_points(capture, offset):
    with np.load(capture) as data:
        depth=data['depth']*float(data['depth_scale']);fx,fy,cx,cy=data['K']
    yy,xx=np.indices(depth.shape)
    camera=np.stack(((xx-cx)*depth/fx,(yy-cy)*depth/fy,depth),axis=-1)
    t=base_to_optical()
    points=(camera@t[:3,:3].T+t[:3,3])*1000+np.asarray(offset)
    points[(depth<=0)|~np.isfinite(depth)]=np.nan
    return points


def mask_reference_centers(capture,masks_path,offset):
    """Surface centroids for identity matching, separate from 6D object poses."""
    points=base_depth_points(capture,offset)
    centers={}
    with np.load(masks_path) as data:
        for name in data.files:
            valid=data[name].astype(bool) & np.isfinite(points).all(axis=-1)
            if valid.sum()>=100:centers[name]=np.median(points[valid,:2],axis=0).tolist()
    return centers


def reference_distance(center, obj, action, reference_xy_mm=None, tolerance_mm=50.):
    start=(np.asarray(obj['base_center_m'][:2])*1000 if reference_xy_mm is None else np.asarray(reference_xy_mm,float))
    delta=center-start
    if action and action['kind']=='knock' and action['target']==obj['name']:
        segment=np.asarray(action['base_direction'][:2])*action['intended_slide_mm']
        fraction=np.clip(delta@segment/max(segment@segment,1e-9),0,1)
        return float(np.linalg.norm(delta-fraction*segment)),tolerance_mm
    return float(np.linalg.norm(delta)),tolerance_mm


def removal_check(capture,points,obj,offset,previous_capture=None):
    import cv2
    center=np.asarray(obj['base_center_m'])*1000
    c,s=np.cos(obj['yaw_rad']),np.sin(obj['yaw_rad'])
    rot=np.array([[c,-s],[s,c]])
    half=np.asarray(obj['dimensions_m'][:2])*500*.6
    corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*half
    xy=corners@rot.T+center[:2]
    table=center[2]-obj['dimensions_m'][2]*500
    # Project the old box's interior top, where valid depth must show clear
    # table after removal; never infer removal from a vanishing SAM track.
    base=np.column_stack((xy,np.full(4,table+obj['dimensions_m'][2]*1000)))
    t=base_to_optical();camera=((base-np.asarray(offset))/1000-t[:3,3])@t[:3,:3]
    with np.load(capture) as data:fx,fy,cx,cy=data['K']
    if np.any(camera[:,2]<=0):return dict(confirmed=False,reason='reference behind camera')
    uv=np.column_stack((fx*camera[:,0]/camera[:,2]+cx,fy*camera[:,1]/camera[:,2]+cy))
    patch=np.zeros(points.shape[:2],np.uint8)
    cv2.fillConvexPoly(patch,np.round(uv).astype(np.int32),1)
    z=points[...,2][patch.astype(bool)]-table
    valid=np.isfinite(z);count=len(z)
    valid_fraction=float(valid.mean()) if count else 0.
    clear_fraction=float(np.mean(np.abs(z[valid])<=7)) if valid.any() else 0.
    drop_fraction=0.
    if previous_capture is not None:
        previous=base_depth_points(previous_capture,offset)[...,2][patch.astype(bool)]-table
        paired=valid & np.isfinite(previous)
        drop_fraction=float(np.count_nonzero(paired & (previous-z>=obj['dimensions_m'][2]*1000*.6))/max(count,1))
    return dict(confirmed=bool(count>=100 and valid_fraction>=.8 and (clear_fraction>=.8 or drop_fraction>=.8)),
                height_drop_fraction=drop_fraction,
                reference_pixels=count,valid_fraction=valid_fraction,table_fraction=clear_fraction,
                table_base_z_mm=float(table),method='table_visible_or_box_height_drop_at_previous_source_footprint')


def associate_masks(capture,masks_path,objects,workflow,offset,action=None,previous_capture=None,mask_references=None):
    """Return one-to-one active-name masks and diagnostic association evidence."""
    points=base_depth_points(capture,offset)
    with np.load(masks_path) as data:masks={n:data[n].astype(bool) for n in data.files}
    lo=np.asarray(workflow['source_xy_min_mm']);hi=np.asarray(workflow['source_xy_max_mm'])
    source=np.all((points[...,:2]>=lo)&(points[...,:2]<=hi),axis=-1)
    table=float(np.median([o['base_center_m'][2]*1000-o['dimensions_m'][2]*500 for o in objects])) if objects else 0.
    candidates=[];discarded=[]
    for name,mask in masks.items():
        valid=mask & np.isfinite(points).all(axis=-1)
        inside=valid & source
        if inside.sum()<100 or inside.sum()<.6*max(valid.sum(),1):
            discarded.append(dict(track=name,reason='outside_source_or_insufficient_depth'));continue
        if np.median(points[inside,2])-table<8:
            discarded.append(dict(track=name,reason='table_surface_not_a_box'));continue
        center=np.median(points[inside,:2],axis=0)
        # The same physical mask must not be assigned twice under two labels.
        if any(np.count_nonzero(mask & m)/max(1,min(mask.sum(),m.sum()))>.5 for _,m,_ in candidates):
            discarded.append(dict(track=name,reason='duplicate_mask'));continue
        candidates.append((name,mask,center))
    n=len(objects);m=len(candidates)
    cost=np.full((n,m+n),1e6)
    distances={};comparisons=[]
    tolerance=float(workflow.get('identity_match_tolerance_mm',50.))
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('identity_match_tolerance_mm must be finite and positive')
    references=mask_references or {}
    for i,obj in enumerate(objects):
        cost[i,m+i]=100.
        for j,(_,_,center) in enumerate(candidates):
            reference=references.get(obj['name'])
            distance,limit=reference_distance(center,obj,action,reference,tolerance)
            comparisons.append(dict(name=obj['name'],tracker_label=candidates[j][0],
                center_base_xy_mm=center.tolist(),reference_distance_mm=distance,limit_mm=limit,
                reference_kind='previous_mask_centroid' if reference is not None else 'object_pose_fallback',
                accepted_distance=distance<=limit))
            distances[i,j]=distance
            if distance<=limit:cost[i,j]=distance/limit
    assigned={};matches=[]
    if n:
        rows,cols=linear_sum_assignment(cost)
        for i,j in zip(rows,cols):
            if j<m and cost[i,j]<=1:
                name=objects[i]['name'];track,mask,center=candidates[j]
                assigned[name]=mask
                matches.append(dict(name=name,tracker_label=track,center_base_xy_mm=center.tolist(),
                                    reference_distance_mm=distances[i,j]))
    removal={}
    if action and action['kind']=='pick':
        obj=next(o for o in objects if o['name']==action['target'])
        removal=removal_check(capture,points,obj,offset,previous_capture)
        if removal['confirmed'] and len(candidates)>=len(objects):
            removal.update(confirmed=False,reason='source_still_contains_the_full_active_box_count')
        if removal['confirmed']:assigned.pop(obj['name'],None)
    return assigned,dict(matches=matches,discarded=discarded,comparisons=comparisons,match_tolerance_mm=tolerance,
                        missing=sorted({o['name'] for o in objects}-set(assigned)),removal=removal)
