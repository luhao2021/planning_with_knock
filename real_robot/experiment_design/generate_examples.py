#!/usr/bin/env python3
"""Generate reference drawings and inventory templates. No camera/robot access."""
import copy
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'real_robot'))
from object_catalog import DIMENSIONS
COLORS = {'protex': '#3b82c4', 'beach': '#e6a044'}


def scene(number, title, entries):
    counts = {'protex': 0, 'beach': 0}
    boxes = []
    for kind, x, y, yaw in entries:
        counts[kind] += 1
        boxes.append(dict(name=f'{kind}_{counts[kind]}', kind=kind, x_mm=x, y_mm=y, yaw_deg=yaw))
    return dict(id=f'S{number:02d}', title=title, boxes=boxes)


def examples():
    p, b = 'protex', 'beach'
    return [
        scene(1, 'Two Protex', [(p,410,-65,15),(p,485,65,100)]),
        scene(2, 'Two BEACH', [(b,410,-75,90),(b,490,75,60)]),
        scene(3, 'One Protex + two BEACH: aligned', [(p,450,62,90),(b,415,-52,90),(b,488,-52,90)]),
        scene(4, 'One BEACH + two Protex: rotated', [(p,390,-85,25),(p,490,0,110),(b,420,110,65)]),
        scene(5, 'Three Protex', [(p,416,-49,90),(p,484,-49,90),(p,450,53,90)]),
        scene(6, 'Three BEACH', [(b,410,-105,75),(b,490,10,105),(b,410,130,65)]),
        scene(7, 'Two BEACH + two Protex', [(b,489,-58,90),(b,489,58,90),(p,416,-51,90),(p,416,51,90)]),
        scene(8, 'One BEACH + three Protex', [(b,489,65,90),(p,416,-51,90),(p,483,-51,90),(p,416,55,90)]),
        scene(9, 'Three BEACH + one Protex', [(b,489,-58,90),(b,416,58,90),(b,489,58,90),(p,416,-58,90)]),
        scene(10, 'Four BEACH', [(b,x,y,90) for x,y in [(416,-58),(489,-58),(416,58),(489,58)]]),
        scene(11, 'Four Protex', [(p,x,y,90) for x,y in [(417,-49),(483,-49),(417,49),(483,49)]]),
        scene(12, 'Three Protex + two BEACH', [(p,416,-51,90),(p,416,51,90),(p,416,153,90),(b,489,-58,90),(b,489,58,90)]),
        scene(13, 'Four Protex + two BEACH', [(p,416,-51,90),(p,416,51,90),(p,416,153,90),(p,365,-165,0),(b,489,-58,90),(b,489,58,90)]),
    ]


def corners(box):
    length, width, _ = np.array(DIMENSIONS[box['kind']]) * 1000
    angle = np.deg2rad(box['yaw_deg'])
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    local = np.array([[-length,-width],[length,-width],[length,width],[-length,width]]) / 2
    return local @ rotation.T + [box['x_mm'],box['y_mm']]


def overlaps(a, b):
    for poly in (a,b):
        for edge in np.roll(poly,-1,axis=0)-poly:
            normal = np.array([-edge[1],edge[0]])
            pa,pb = a @ normal,b @ normal
            if pa.max() <= pb.min()+1e-6 or pb.max() <= pa.min()+1e-6:
                return False
    return True


def validate(scenes):
    for item in scenes:
        if not 2 <= len(item['boxes']) <= 6:
            raise ValueError(f"{item['id']}: expected two to six boxes")
        polygons = [corners(box) for box in item['boxes']]
        for index, poly in enumerate(polygons):
            if np.any(poly.min(axis=0) < [300,-250]) or np.any(poly.max(axis=0) > [570,250]):
                raise ValueError(f"{item['id']}: footprint outside source area")
            if any(overlaps(poly,other) for other in polygons[:index]):
                raise ValueError(f"{item['id']}: overlapping nominal footprints")


def draw(ax, item):
    ax.set(xlim=(285,585), ylim=(-265,265), aspect='equal', xlabel='link_base X (mm)', ylabel='link_base Y (mm)')
    ax.add_patch(Polygon([[300,-250],[570,-250],[570,250],[300,250]],fill=False,edgecolor='#777777',linestyle='--'))
    for box in item['boxes']:
        ax.add_patch(Polygon(corners(box),facecolor=COLORS[box['kind']],edgecolor='#263238',alpha=.85))
        label = box['name'].replace('protex_','P').replace('beach_','B')
        ax.text(box['x_mm'],box['y_mm'],label,ha='center',va='center',fontweight='bold',fontsize=9)
    import textwrap
    ax.set_title(textwrap.fill(f"{item['id']} · {item['title']}", width=34),fontsize=10)
    ax.grid(alpha=.15)


def main():
    scenes = examples()
    validate(scenes)
    (HERE / 'configs').mkdir(exist_ok=True)
    (HERE / 'layouts').mkdir(exist_ok=True)
    base = json.loads((ROOT / 'real_robot/pipeline_config.local.json').read_text())
    for item in scenes:
        config = copy.deepcopy(base)
        config['experiment_reference'] = dict(scene_id=item['id'], title=item['title'], geometry_is_reference_only=True)
        if item['id'] in ('S07','S12','S13'):
            config['execution_overrides']={
                'class_to_bowl':{'protex':'bowl_1','beach':'bowl_2'},
                'release_tcp_z_mm_by_class':{'protex':{'S07':60,'S12':80,'S13':110}[item['id']],'beach':60}}
        perception = config['perception']
        only_protex = all(box['kind']=='protex' for box in item['boxes'])
        perception['automatic_scene']['enabled'] = only_protex
        perception['automatic_scene']['expected_instances'] = len(item['boxes'])
        if len(item['boxes']) > 4:
            # Larger mixed scenes use the manual inventory; automatic Protex stays capped at four.
            perception['automatic_scene'] = {'enabled': False}
            workflow = config['workflow']
            workflow['max_actions'] = max(workflow['max_actions'], len(item['boxes']) *
                (workflow['max_knock_attempts'] + workflow['max_pick_attempts']))
        perception['objects'] = {}
        for box in item['boxes']:
            kind = box['kind']
            size = 'x'.join(f'{value*1000:g}' for value in DIMENSIONS[kind])
            perception['objects'][box['name']] = dict(**{'class':kind},instance_id=box['name'].split('_')[-1],
                mesh=f'real_robot/models/{kind}_{size}mm.obj')
        if not only_protex:
            perception['points'] = f'real_robot/experiment_design/configs/{item["id"]}.NEEDS_SCENE_SETUP.points.json'
        (HERE / 'configs' / f'{item["id"]}.json').write_text(json.dumps(config,indent=2)+'\n')
    (HERE / 'reference_scenes.json').write_text(json.dumps(dict(reference_only=True,units='mm',
        frame='link_base', yaw='degrees counterclockwise from +X along long side',
        nominal_dimensions_mm={k:[v*1000 for v in DIMENSIONS[k]] for k in COLORS},scenes=scenes),indent=2)+'\n')
    with (HERE / 'reference_positions.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,lineterminator='\n',fieldnames=['scene_id','name','kind','x_mm','y_mm','yaw_deg'])
        writer.writeheader()
        for item in scenes:
            for box in item['boxes']:
                writer.writerow(dict(scene_id=item['id'],**box))
    legend = [Patch(facecolor=color,label=f'{kind.capitalize()} ({"90 × 60" if kind=="protex" else "110 × 65"} mm)') for kind,color in COLORS.items()]
    rows = (len(scenes)+3)//4
    fig, axes = plt.subplots(rows,4,figsize=(14,6*rows),layout='constrained')
    for ax,item in zip(axes.flat,scenes): draw(ax,item)
    for ax in list(axes.flat)[len(scenes):]: ax.set_visible(False)
    fig.suptitle('Thirteen physical scene references — arrange approximately; detect actual poses\n'
                 'Blue = Protex · Orange = BEACH · Top view, +X right, +Y up · Dashed = source area',fontsize=14)
    fig.savefig(HERE / 'overview.png',dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,9),layout='constrained')
    focused = [item for item in scenes if item['id'] in ('S07','S12')]
    for ax,item in zip(axes,focused): draw(ax,item)
    fig.suptitle('Focus templates: S07 and S12\nReference only — detect actual positions and orientations',fontsize=13)
    fig.savefig(HERE / 'focus_S07_S12.png',dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,9),layout='constrained')
    focused = [item for item in scenes if item['id'] in ('S07','S12','S13')]
    for ax,item in zip(axes,focused): draw(ax,item)
    fig.suptitle('Focus templates: S07, S12 and S13\nReference only — detect actual positions and orientations',fontsize=13)
    fig.savefig(HERE / 'focus_S07_S12_S13.png',dpi=150)
    plt.close(fig)
    with PdfPages(HERE / 'reference_layouts.pdf') as pdf:
        for item in scenes:
            fig = plt.figure(figsize=(8.27,11.69))
            ax = fig.add_axes([.13,.34,.74,.57])
            draw(ax,item)
            fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2)
            rows = [f"{box['name']:<10}  X={box['x_mm']:6.1f}  Y={box['y_mm']:6.1f}  yaw={box['yaw_deg']:6.1f}°" for box in item['boxes']]
            fig.text(.13,.28,'Nominal reference centres (mm) and long-side yaw:\n'+'\n'.join(rows),fontfamily='monospace',fontsize=11,va='top')
            fig.text(.13,.08,'Reference only: not a robot trajectory or a required detected layout.\n'
                     'Boxes lie flat: Protex height 28 mm; BEACH height 25 mm.\n'
                     'P/B numbers label the drawing; automatic names can differ.',fontsize=10)
            pdf.savefig(fig)
            fig.savefig(HERE / 'layouts' / f'{item["id"]}.png',dpi=120)
            plt.close(fig)
    results = HERE / 'results_template.csv'
    fields = ['scene_id','layout_variant','trial','protex_count','beach_count','run_directory','git_commit','all_boxes_placed',
              'boxes_placed','knock_batches','pick_batches','elapsed_seconds','failure_stage','interventions','notes']
    existing_ids = set()
    if results.exists():
        with results.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            existing_ids = {row['scene_id'] for row in reader}
    with results.open('a',newline='') as handle:
        writer = csv.DictWriter(handle,lineterminator='\n',fieldnames=fields)
        if not existing_ids and results.stat().st_size == 0:
            writer.writeheader()
        for item in scenes:
            if item['id'] not in existing_ids:
                writer.writerow(dict(scene_id=item['id'],layout_variant='A',trial=1,
                    protex_count=sum(b['kind']=='protex' for b in item['boxes']),
                    beach_count=sum(b['kind']=='beach' for b in item['boxes'])))
    print(f'Validated and generated {len(scenes)} non-overlapping references, drawings and inventory templates in {HERE}')


if __name__ == '__main__':
    main()
