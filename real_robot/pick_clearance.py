"""Offline XY clearance for the commanded open-to-closed finger sweep.

Uses nominal box footprints and the configured planning finger thickness. This
checks the two finger contact regions, not the whole gripper/arm or pose accuracy.
"""
import math
import numpy as np
from xarm_executor import UnsafePlan


def rectangle(center, tangent, normal, half_tangent, half_normal):
    return np.asarray([center+a*half_tangent*tangent+b*half_normal*normal
                       for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])


def overlap_depth_mm(a,b):
    """Minimum separating translation for convex rectangles; None if disjoint.

    Vertices are in mm. Unlike area or interval-intersection width, separating
    translation also handles containment and is independent of world-axis yaw.
    """
    depth=float('inf')
    for polygon in (a,b):
        for edge in (polygon[1]-polygon[0],polygon[2]-polygon[1]):
            axis=np.array([-edge[1],edge[0]])
            axis/=np.linalg.norm(axis)
            pa,pb=a@axis,b@axis
            separation=min(float(pa.max()-pb.min()),float(pb.max()-pa.min()))
            if separation < -1e-6:return None
            depth=min(depth,max(0.,separation))
    return depth


def intersects(a,b):
    return overlap_depth_mm(a,b) is not None


def pick_clearance(layout,candidate,cfg):
    obj=next(o for o in layout['objects'] if o['name']==candidate['target'])
    witness=candidate['witness']
    if not witness.startswith(('pick-x@','pick-y@')):raise UnsafePlan('invalid Pick witness')
    orientation=witness[5]
    placement=float(witness.split('@')[1])*1000
    dims=np.asarray(obj['dimensions_m'])*1000
    yaw=float(obj['yaw_rad']);center=np.asarray(obj['base_center_m'][:2])*1000
    xaxis=np.array([math.cos(yaw),math.sin(yaw)]);yaxis=np.array([-math.sin(yaw),math.cos(yaw)])
    tangent,normal=(xaxis,yaxis) if orientation=='y' else (yaxis,xaxis)
    width,length=(dims[1],dims[0]) if orientation=='y' else (dims[0],dims[1])
    thickness=float(layout.get('finger_thickness_mm',20))
    opening=float(cfg['gripper_open_mm'])
    reasons=[]
    if not np.isfinite([placement,width,length,thickness,opening]).all() or thickness<=0:
        raise UnsafePlan('invalid Pick clearance geometry')
    if placement < -length/2-1e-6 or placement+30 > length/2+1e-6:
        reasons.append('finger contact window extends beyond nominal target')
    if width+2>min(opening,cfg['usable_jaw_opening_mm']):
        reasons.append('nominal target exceeds usable open aperture')
    grasp=center+(placement+15)*tangent
    from gripper_command import pick_aperture_mm
    # Invalid apertures already fail the batch; retain diagnostic output here.
    try:aperture=pick_aperture_mm(cfg,obj['kind'],width)
    except UnsafePlan as exc:
        reasons.append(str(exc));aperture=width
    inner=aperture/2
    outer=opening/2+thickness
    sweeps=[rectangle(grasp+sign*(inner+outer)/2*normal,tangent,normal,15,(outer-inner)/2)
            for sign in (-1,1)]
    tolerance=float(cfg.get('pick_overlap_tolerance_mm',0))
    if not np.isfinite(tolerance) or not 0<=tolerance<=5:
        raise UnsafePlan('pick_overlap_tolerance_mm must be finite and between 0 and 5')
    blockers=[];tolerated=[];overlaps=[]
    for other in layout['objects']:
        if other['name']==obj['name']:continue
        theta=float(other['yaw_rad']);u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-math.sin(theta),math.cos(theta)])
        nominal=rectangle(np.asarray(other['base_center_m'][:2])*1000,u,v,
                          other['dimensions_m'][0]*500,other['dimensions_m'][1]*500)
        depths=[depth for sweep in sweeps if (depth:=overlap_depth_mm(sweep,nominal)) is not None]
        if depths:
            depth=max(depths)
            accepted=tolerance>0 and depth<=tolerance+1e-6
            overlaps.append(dict(name=other['name'],estimated_overlap_mm=depth,tolerated=accepted))
            (tolerated if accepted else blockers).append(other['name'])
    if blockers:reasons.append('open-to-closed finger sweep intersects neighboring boxes')
    return dict(passed=not reasons,reasons=reasons,blockers=blockers,
                overlap_tolerance_mm=tolerance,tolerated_neighbors=tolerated,overlap_details=overlaps,
                finger_thickness_mm=thickness,open_aperture_mm=opening,
                nominal_closing_width_mm=float(width),commanded_close_aperture_estimate_mm=aperture,finger_window_mm=30,
                swept_finger_polygons_base_mm=[p.tolist() for p in sweeps],
                pose_source='observed_objects_before_planning_overlap_correction',
                box_dimensions='nominal_unshrunk')
