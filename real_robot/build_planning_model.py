"""Build a PyBullet URDF from official xArm7/G1 assets; no hardware access."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import yaml

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTION=ROOT/'real_robot/external/xarm_ros2/xarm_description'
OUT=ROOT/'real_robot/models/xarm7_planning'
LIMITS=[(-6.283185307,6.283185307),(-2.059,2.0944),(-6.283185307,6.283185307),(-.19198,3.927),(-6.283185307,6.283185307),(-1.69297,3.141592654),(-6.283185307,6.283185307)]


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    kin_path=DESCRIPTION/'config/kinematics/default/xarm7_default_kinematics.yaml'
    kin=yaml.safe_load(kin_path.read_text())['kinematics']
    robot=ET.Element('robot',name='xarm7_g1_planning')
    def link(name,mesh=None):
        e=ET.SubElement(robot,'link',name=name)
        inertial=ET.SubElement(e,'inertial');ET.SubElement(inertial,'mass',value='1')
        ET.SubElement(inertial,'inertia',ixx='.01',iyy='.01',izz='.01',ixy='0',ixz='0',iyz='0')
        if mesh:
            if not mesh.exists():raise FileNotFoundError(mesh)
            for tag in ('visual','collision'):
                g=ET.SubElement(ET.SubElement(e,tag),'geometry');ET.SubElement(g,'mesh',filename=str(mesh))
        return e
    def joint(name,parent,child,xyz,rpy,axis=None,limits=None):
        e=ET.SubElement(robot,'joint',name=name,type='revolute' if axis else 'fixed')
        ET.SubElement(e,'parent',link=parent);ET.SubElement(e,'child',link=child)
        ET.SubElement(e,'origin',xyz=xyz,rpy=rpy)
        if axis:
            ET.SubElement(e,'axis',xyz=axis)
            ET.SubElement(e,'limit',lower=str(limits[0]),upper=str(limits[1]),effort='50',velocity='1')
        return e
    meshroot=DESCRIPTION/'meshes'
    link('link_base',meshroot/'xarm7/visual/link_base.stl')
    for i in range(1,8):
        mesh=meshroot/(f'xarm7/visual/link{i}.stl' if i<7 else 'end_tool/collision/end_tool_1300.stl')
        link(f'link{i}',mesh);k=kin[f'joint{i}']
        joint(f'joint{i}','link_base' if i==1 else f'link{i-1}',f'link{i}',
              ' '.join(str(k[v]) for v in ('x','y','z')),
              ' '.join(str(k[v]) for v in ('roll','pitch','yaw')),'0 0 1',LIMITS[i-1])
    # Copy the official G1 mesh/joint topology. PyBullet mimic joints are driven explicitly.
    gripper=ET.parse(DESCRIPTION/'urdf/gripper/xarm_gripper.urdf.xacro').getroot()
    macro=next(iter(gripper))
    for src in macro.findall('link'):
        name=src.get('name').replace('${prefix}','')
        if name=='link_tcp':continue
        meshname='base_link' if name=='xarm_gripper_base_link' else name
        link(name,meshroot/f'gripper/xarm/{meshname}.stl')
    joint('gripper_fix','link7','xarm_gripper_base_link','0 0 0','0 0 0')
    for src in macro.findall('joint'):
        name=src.get('name').replace('${prefix}','')
        if src.get('type')=='fixed':continue
        org=src.find('origin');axis=src.find('axis');lim=src.find('limit')
        joint(name,src.find('parent').get('link').replace('${prefix}',''),src.find('child').get('link').replace('${prefix}',''),
              org.get('xyz','0 0 0'),org.get('rpy','0 0 0'),axis.get('xyz'),(float(lim.get('lower')),float(lim.get('upper'))))
    link('configured_tcp');joint('configured_tcp_fixed','link7','configured_tcp','0 0 .172','0 0 0')
    ET.indent(robot);path=OUT/'xarm7_g1.urdf';ET.ElementTree(robot).write(path,encoding='unicode')
    srdf=ET.parse(ROOT/'real_robot/external/xarm_ros2/xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro')
    exclusions=[]
    for e in srdf.iter('disable_collisions'):
        a,b=(e.get(k).replace('${prefix}','') for k in ('link1','link2'))
        exclusions.append([a,b])
    metadata={'source':'UFACTORY xarm_ros2 humble','robot_variant':'xArm7 model1300 with G1 gripper',
              'kinematics':'official nominal; check against controller FK before motion',
              'joint_origins':kin,'joint_limits_rad':LIMITS,'disabled_self_collision_pairs':exclusions}
    (OUT/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path

if __name__=='__main__':print(build())
