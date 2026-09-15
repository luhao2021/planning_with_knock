"""Conservative initial Pick rules, independent of execution acceptance checks.

A solid rectangle covering the full open gripper plus a planning margin contains
both finger sweeps throughout closure. Nominal observed boxes (not corrected or
shrunk footprints) are obstacles. The shared solver and Isaac rules are unchanged.
"""
import math
import numpy as np
from cuboid_knock_pick_solvers import (Action,RuleInstance,Rect,nondominated_actions,
                                      knock_blockers,_clip_box)


def planning_pick_actions(inst,index,obj,observed,model):
    opening=model['open_aperture_mm']/1000
    thickness=model['finger_thickness_mm']/1000
    margin=model['margin_mm']/1000
    usable=model['usable_jaw_opening_mm']/1000
    shrink=model['footprint_shrink_mm']/1000
    if not np.isfinite([opening,thickness,margin,usable,shrink]).all() or min(opening,thickness,usable)<=0 or margin<0:
        raise ValueError('invalid conservative planning gripper geometry')
    yaw=float(obj['yaw_rad']);c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.array([[c,-s],[s,c]])
    center=np.asarray(obj['base_center_m'][:2])
    length,width=obj['dimensions_m'][:2]
    outer=opening/2+thickness+margin
    candidates={}
    for orientation,closing,tangent_length in [('x',length,width),('y',width,length)]:
        if closing+.002>min(opening,usable)+1e-10:continue
        lo=-(tangent_length-shrink)/2;hi=(tangent_length-shrink)/2-inst.F
        if hi<lo:continue
        contacts={}
        for j,other in enumerate(observed):
            if j==index:continue
            ox,oy=other['base_center_m'][:2];ol,ow=other['dimensions_m'][:2]
            rect=Rect(other['name'],ox-ol/2,ox+ol/2,oy-ow/2,oy+ow/2,other['yaw_rad'])
            local=(rect.corners-center)@rotation
            if orientation=='y':
                clipped=_clip_box(local,lo-margin,hi+inst.F+margin,-outer,outer);axis=0
            else:
                clipped=_clip_box(local,-outer,outer,lo-margin,hi+inst.F+margin);axis=1
            if len(clipped):contacts[j]=(float(clipped[:,axis].min()),float(clipped[:,axis].max()))
        critical={lo,hi}
        for a,b in contacts.values():
            for t in (a-inst.F-margin,b+margin):
                if lo<=t<=hi:critical.add(t)
        points=sorted(critical)
        points=sorted(set(points+[(a+b)/2 for a,b in zip(points,points[1:])]))
        for t in points:
            # The witness is the execution contract. Evaluate the rounded value
            # that build_batch will actually parse, not a higher-precision point.
            witness=f'pick-{orientation}@{t:.9f}'
            t=float(witness.split('@')[1])
            blockers=frozenset(j for j,(a,b) in contacts.items()
                              if min(t+inst.F+margin,b)-max(t-margin,a)>=-1e-9)
            candidates.setdefault(blockers,Action(index,'pick',blockers,0,witness,
                                                  orientation=orientation,placement=t))
    return [a for blockers,a in candidates.items()
            if not any(other<blockers for other in candidates)]


def conservative_rule(inst,observed_objects,model):
    by_name={o['name']:o for o in observed_objects}
    if len(by_name)!=len(observed_objects) or set(by_name)!={r.name for r in inst.rects}:
        raise ValueError('planning geometry and observed object identities differ')
    observed=[by_name[r.name] for r in inst.rects]
    actions={};picks={}
    for i,obj in enumerate(observed):
        picks[i]=planning_pick_actions(inst,i,obj,observed,model)
        knocks=[Action(i,'knock',knock_blockers(inst,i,d),1,'knock-'+d,direction=d)
                for d in ('+x','-x','+y','-y')]
        actions[i]=nondominated_actions(picks[i]+knocks)
    return RuleInstance(inst,actions),picks
