"""Generate Protex scene seeds from RGB-D and automatic SAM2 masks.

No camera or SDK access. Masks determine the count; an optional expected count
checks the configured inventory. This mode assumes visible, flat Protex boxes.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from pose_to_plan import base_to_optical,checked_base_pose_offset_mm
from xarm_executor import UnsafePlan


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def depth_workspace(capture,workflow,offset_mm):
    with np.load(capture) as data:
        z=np.asarray(data['depth'],float)*float(data['depth_scale'])
        fx,fy,cx,cy=np.asarray(data['K'],float)
        rgb=data['color'].copy()
    if z.shape!=rgb.shape[:2] or not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy)<=0:
        raise UnsafePlan('invalid RGB-D/intrinsics for automatic scene initialization')
    v,u=np.indices(z.shape)
    camera=np.stack(((u-cx)*z/fx,(v-cy)*z/fy,z),-1)
    transform=base_to_optical()
    base=camera@transform[:3,:3].T+transform[:3,3]+np.asarray(checked_base_pose_offset_mm(offset_mm))/1000
    lo=np.asarray(workflow['source_xy_min_mm'])/1000;hi=np.asarray(workflow['source_xy_max_mm'])/1000
    workspace=(z>0)&np.isfinite(base).all(axis=2)&np.all(base[:,:,:2]>=lo,axis=2)&np.all(base[:,:,:2]<=hi,axis=2)
    if np.count_nonzero(workspace)<500:
        raise UnsafePlan('insufficient valid source-area depth for automatic scene initialization')
    return rgb,base,workspace


def prepare_grid(capture,workflow,offset_mm,folder,*,spacing_px=5):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rgb,base,workspace=depth_workspace(capture,workflow,offset_mm)
    raised=workspace&(base[:,:,2]>.018)&(base[:,:,2]<.065)
    v,u=np.indices(raised.shape)
    if not isinstance(spacing_px,int) or spacing_px<1:raise ValueError("grid spacing must be a positive integer")
    ys,xs=np.where(raised&(u%spacing_px==0)&(v%spacing_px==0))
    grid=np.column_stack((xs/raised.shape[1],ys/raised.shape[0])).astype(np.float32)
    path=folder/'point_grid.npz';np.savez_compressed(path,point_grid=grid)
    cv2.imwrite(str(folder/'depth_prompt_region.png'),raised.astype(np.uint8)*255)
    write(folder/'depth_prompt_grid.json',dict(points=len(grid),spacing_px=spacing_px,height_band_base_mm=[18,65],
                                              source_xy_min_mm=workflow['source_xy_min_mm'],source_xy_max_mm=workflow['source_xy_max_mm'],
                                              base_pose_offset_mm=offset_mm))
    return path,len(grid)


def largest_component(mask):
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if count<=1:return np.zeros_like(mask,dtype=bool)
    return labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))


def select_instances(capture,workflow,offset_mm,masks,records,folder,*,max_instances=4,expected_instances=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rgb,base,workspace=depth_workspace(capture,workflow,offset_mm)
    candidates=[];rejected=[]
    for record in records:
        name=record['name'];original=np.asarray(masks[name],bool)
        if original.shape!=workspace.shape:raise UnsafePlan('automatic mask shape differs from capture')
        mask=largest_component(original)
        area=int(mask.sum())
        if area<250:
            rejected.append(dict(name=name,reason='mask too small'));continue
        if np.count_nonzero(mask&workspace)/area<.85:
            rejected.append(dict(name=name,reason='outside source workspace or missing depth'));continue
        interior=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        xyz=base[interior&workspace]
        if len(xyz)<100:continue
        height=float(np.median(xyz[:,2]))
        if not .020<height<.055:
            rejected.append(dict(name=name,reason='not at a flat Protex top height',median_height_mm=height*1000));continue
        xyz=xyz[(xyz[:,2]>height-.012)&(xyz[:,2]<height+.010)]
        lo,hi=np.quantile(xyz[:,:2],[.02,.98],axis=0)
        xy=xyz[np.all((xyz[:,:2]>=lo)&(xyz[:,:2]<=hi),axis=1),:2]
        if len(xy)<100:continue
        rect=cv2.minAreaRect(xy.astype(np.float32));short,long=sorted(rect[1])
        metric=dict(name=name,median_height_mm=height*1000,visible_footprint_mm=[short*1000,long*1000])
        if not (.044<short<.080 and .068<long<.108):
            rejected.append(dict(**metric,reason='fragment/merged mask or unsupported dimensions'));continue
        # Rank full-box masks; remove duplicate proposals by shared pixels.
        quality=float(record['predicted_iou'])+float(record['stability_score'])
        quality-=abs(short-.060)/.060+abs(long-.084)/.090
        candidates.append(dict(metric=metric,mask=mask,quality=quality,center=list(rect[0])))
    chosen=[]
    for item in sorted(candidates,key=lambda c:c['quality'],reverse=True):
        duplicate=any(np.count_nonzero(item['mask']&prev['mask'])/min(item['mask'].sum(),prev['mask'].sum())>.35 for prev in chosen)
        if duplicate:
            rejected.append(dict(**item['metric'],reason='duplicate/overlapping proposal'));continue
        chosen.append(item)
    if len(chosen)>max_instances:
        write(folder/'selection.json',dict(selected=[c['metric'] for c in chosen],rejected=rejected))
        raise UnsafePlan(f'automatic scene found {len(chosen)} candidate boxes, above configured inventory {max_instances}')
    union=np.zeros(workspace.shape,bool)
    for item in chosen:union|=item['mask']
    # Unexplained raised regions can indicate a missed object. Do not silently
    # infer fewer boxes from incomplete segmentation. Ignore isolated depth noise.
    raised=workspace&(base[:,:,2]>.022)&(base[:,:,2]<.065)
    covered=cv2.dilate(union.astype(np.uint8),np.ones((7,7),np.uint8))>0
    unexplained=raised&~covered
    count,_,stats,_=cv2.connectedComponentsWithStats(unexplained.astype(np.uint8),8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0
    report=dict(mode='automatic_protex',count=len(chosen),selected=[c['metric'] for c in chosen],
                rejected=rejected,unexplained_raised_pixels=int(unexplained.sum()),largest_unexplained_region_pixels=largest)
    write(folder/'selection.json',report)
    cv2.imwrite(str(folder/'unexplained_raised_regions.png'),unexplained.astype(np.uint8)*255)
    if expected_instances is not None and len(chosen)!=expected_instances:
        raise UnsafePlan(f'automatic scene found {len(chosen)} boxes; expected {expected_instances}')
    if expected_instances is None and (largest>250 or unexplained.sum()>max(400,.20*int(raised.sum()))):
        raise UnsafePlan('automatic scene has unexplained raised regions; refusing an incomplete object inventory')
    # Rows in image coordinates, then left-to-right. Names are fresh per run.
    for item in chosen:
        ys,xs=np.where(item['mask']);item['pixel_center']=[float(xs.mean()),float(ys.mean())]
    chosen.sort(key=lambda c:(round(c['pixel_center'][1]/25),c['pixel_center'][0]))
    points={};selected_masks={};overlay=rgb.copy()
    for index,item in enumerate(chosen,1):
        name=f'protex_{index}'
        distance=cv2.distanceTransform((item['mask']&raised).astype(np.uint8),cv2.DIST_L2,5)
        v,u=np.unravel_index(np.argmax(distance),distance.shape)
        if distance[v,u]<2:raise UnsafePlan('automatic box mask has no reliable interior seed')
        points[name]=[int(u),int(v)];selected_masks[name]=item['mask']
        color=(50,230,60);cv2.drawContours(overlay,cv2.findContours(item['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,color,1)
        cv2.circle(overlay,(int(u),int(v)),4,(0,0,255),-1)
        cv2.putText(overlay,name,(int(u)+5,int(v)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
    write(folder/'generated_points.json',points)
    np.savez_compressed(folder/'selected_masks.npz',**selected_masks)
    cv2.imwrite(str(folder/'automatic_seeds.png'),overlay)
    return points
