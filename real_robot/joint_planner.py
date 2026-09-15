"""Offline xArm7 joint-space planning with RRT-Connect and mesh collision checks.

No SDK import or robot connection. All units here are metres and radians.
"""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from xarm_executor import UnsafePlan

ROOT=Path(__file__).resolve().parents[1]


def pose_matrix(pose):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',pose[3:],degrees=True).as_matrix();t[:3,3]=np.asarray(pose[:3])/1000
    return t


class RobotModel:
    def __init__(self,scene=(),*,margin_mm=2,table_z_mm=0):
        import pybullet as p
        self.p=p;self.client=p.connect(p.DIRECT)
        model_dir=ROOT/'real_robot/models/xarm7_planning'
        if not (model_dir/'xarm7_g1.urdf').exists():
            from build_planning_model import build
            build()
        self.metadata=json.loads((model_dir/'model.json').read_text())
        self.robot=p.loadURDF(str(model_dir/'xarm7_g1.urdf'),useFixedBase=True,
                             flags=p.URDF_USE_SELF_COLLISION,physicsClientId=self.client)
        self.names={-1:'link_base'};self.arm=[];self.gripper=[]
        for i in range(p.getNumJoints(self.robot,physicsClientId=self.client)):
            info=p.getJointInfo(self.robot,i,physicsClientId=self.client);self.names[i]=info[12].decode()
            name=info[1].decode()
            if name in [f'joint{k}' for k in range(1,8)]:self.arm.append(i)
            elif info[2]==p.JOINT_REVOLUTE:self.gripper.append(i)
            if self.names[i]=='configured_tcp':self.tcp_index=i
        self.limits=np.array(self.metadata['joint_limits_rad'])
        self.disabled={frozenset(x) for x in self.metadata['disabled_self_collision_pairs']}
        self.margin=margin_mm/1000;self.obstacles={};self.carried=None;self.carried_relative=None
        self.table_z=table_z_mm/1000
        # Table plane for all links except the anchored base/first link.
        self.table=self.box([2,2,.1],[.5,0,self.table_z-.05],0)
        for obj in scene:self.obstacles[obj['name']]=self.box(obj['dimensions_m'],obj['base_center_m'],obj['yaw_rad'])
        self.set_gripper(85)
        self.last_collision=None
        self._origins=[]
        for i in range(1,8):
            k=self.metadata['joint_origins'][f'joint{i}'];t=np.eye(4)
            t[:3,:3]=Rotation.from_euler('xyz',[k[x] for x in ('roll','pitch','yaw')]).as_matrix()
            t[:3,3]=[k[x] for x in ('x','y','z')];self._origins.append(t)

    def close(self):self.p.disconnect(self.client)

    def box(self,dims,center,yaw):
        p=self.p;shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=(np.asarray(dims)/2).tolist(),physicsClientId=self.client)
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=center,
                                 baseOrientation=Rotation.from_euler('z',yaw).as_quat(),physicsClientId=self.client)

    def set_gripper(self,opening_mm):
        # Nominal G1 parallel finger kinematics, with the mesh's inner-face offset.
        # Keep a full-open geometric model if a requested aperture is uncalibrated.
        q=float(np.clip(.85*(1-opening_mm/85),0,.85))
        for i in self.gripper:self.p.resetJointState(self.robot,i,q,physicsClientId=self.client)

    def fk(self,q):
        t=np.eye(4)
        for origin,angle in zip(self._origins,q):
            c,s=math.cos(angle),math.sin(angle)
            rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
            t=t@origin@rz
        tcp=np.eye(4);tcp[2,3]=.172
        return t@tcp

    def set_q(self,q):
        for i,v in zip(self.arm,q):self.p.resetJointState(self.robot,i,float(v),physicsClientId=self.client)
        if self.carried is not None:
            t=self.fk(q)@self.carried_relative
            self.p.resetBasePositionAndOrientation(self.carried,t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat(),physicsClientId=self.client)

    def attach(self,obj,grasp_pose):
        if obj['name'] in self.obstacles:self.p.removeBody(self.obstacles.pop(obj['name']),physicsClientId=self.client)
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',obj['yaw_rad']).as_matrix();t[:3,3]=obj['base_center_m']
        self.carried_relative=np.linalg.inv(pose_matrix(grasp_pose))@t
        self.carried=self.box(obj['dimensions_m'],[0,0,1],0)
        self.carried_dims=np.asarray(obj['dimensions_m']);self.carried_name=obj['name']

    def release(self,drop):
        self.p.removeBody(self.carried,physicsClientId=self.client);self.carried=None
        self.obstacles[self.carried_name]=self.box(self.carried_dims,[drop[0]/1000,drop[1]/1000,self.table_z+self.carried_dims[2]/2],0)

    def valid(self,q,*,support_contact=False):
        q=np.asarray(q)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<self.limits[:,0]) or np.any(q>self.limits[:,1]):return False
        self.set_q(q);p=self.p
        tcp=self.fk(q)
        if tcp[2,3]<.010:return False
        for c in p.getClosestPoints(self.robot,self.robot,self.margin,physicsClientId=self.client):
            a,b=c[3:5]
            if a==b or frozenset((self.names[a],self.names[b])) in self.disabled:continue
            self.last_collision=('self',self.names[a],self.names[b],c[8]);return False
        for label,body in [('table',self.table),*self.obstacles.items()]:
            for c in p.getClosestPoints(self.robot,body,self.margin,physicsClientId=self.client):
                if label=='table' and c[3] in (-1,self.arm[0]):continue
                self.last_collision=(label,self.names[c[3]],c[8]);return False
        if self.carried is not None:
            for c in p.getClosestPoints(self.robot,self.carried,self.margin,physicsClientId=self.client):
                if self.names[c[3]] in ('left_finger','right_finger'):continue
                self.last_collision=('carried',self.names[c[3]],c[8]);return False
            for label,body in [('table',self.table),*self.obstacles.items()]:
                for c in p.getClosestPoints(self.carried,body,self.margin,physicsClientId=self.client):
                    if label=='table' and support_contact and c[8]>=-.005:continue
                    self.last_collision=('carried',label,c[8]);return False
        self.last_collision=None;return True

    def ik(self,pose,seed,*,tries=16,require_collision_free=True,support_contact=False,rng=None):
        goal=pose_matrix(pose);rng=rng or np.random.default_rng(11)
        def residual(q):
            t=self.fk(q)
            return np.r_[5*(t[:3,3]-goal[:3,3]),Rotation.from_matrix(t[:3,:3]@goal[:3,:3].T).as_rotvec()]
        for attempt in range(tries):
            start=np.array(seed) if attempt==0 else (np.array(seed)+rng.normal(0,.45,7) if attempt<8 else rng.uniform(self.limits[:,0],self.limits[:,1]))
            start=np.clip(start,self.limits[:,0]+1e-7,self.limits[:,1]-1e-7)
            result=least_squares(residual,start,bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=130,
                                 ftol=1e-8,xtol=1e-8,gtol=1e-8)
            err=residual(result.x)
            if np.linalg.norm(err[:3])/5>.0008 or np.linalg.norm(err[3:])>np.deg2rad(.4):continue
            if not require_collision_free or self.valid(result.x,support_contact=support_contact):return result.x
        raise UnsafePlan(f'no collision-free IK for {pose}; last collision={self.last_collision}')


class RRTConnect:
    def __init__(self,valid,limits,*,seed=11,step_deg=12,resolution_deg=.75,max_iterations=5000,timeout_s=30):
        self.valid=valid;self.limits=np.asarray(limits);self.rng=np.random.default_rng(seed)
        self.step=np.deg2rad(step_deg);self.resolution=np.deg2rad(resolution_deg)
        self.max_iterations=max_iterations;self.timeout=timeout_s;self.checks=0

    def edge(self,a,b):
        for t in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/self.resolution))+1)):
            self.checks+=1
            if not self.valid(a+t*(b-a)):return False
        return True

    @staticmethod
    def trace(tree,index):
        path=[]
        while index!=-1:
            q,index=tree[index];path.append(q)
        return path[::-1]

    def plan(self,start,goal):
        start,goal=np.asarray(start),np.asarray(goal)
        if not self.valid(start) or not self.valid(goal):raise UnsafePlan('RRT start or goal collides')
        trees=[[(start,-1)],[(goal,-1)]];began=time.monotonic()
        def extend(tree,target):
            i=int(np.argmin([np.linalg.norm(q-target) for q,_ in tree]));q=tree[i][0]
            distance=np.linalg.norm(target-q);new=target.copy() if distance<=self.step else q+(target-q)*self.step/distance
            if not self.edge(q,new):return None,False
            tree.append((new,i));return len(tree)-1,distance<=self.step
        for iteration in range(self.max_iterations):
            if time.monotonic()-began>self.timeout:break
            side=iteration%2;a,b=trees[side],trees[1-side]
            target=b[0][0] if self.rng.random()<.2 else self.rng.uniform(self.limits[:,0],self.limits[:,1])
            ia,_=extend(a,target)
            if ia is None:continue
            while time.monotonic()-began<=self.timeout:
                ib,reached=extend(b,a[ia][0])
                if ib is None:break
                if reached:
                    pa,pb=self.trace(a,ia),self.trace(b,ib)
                    path=pa+pb[-2::-1] if side==0 else pb+pa[-2::-1]
                    # Shortcut only checked joint-space edges; never Cartesian interpolation.
                    for _ in range(40):
                        if len(path)<3:break
                        i,j=sorted(self.rng.choice(len(path),2,replace=False))
                        if j>i+1 and self.edge(path[i],path[j]):path=path[:i+1]+path[j:]
                    if not all(self.edge(x,y) for x,y in zip(path,path[1:])):raise UnsafePlan('RRT path failed final edge check')
                    return path,dict(algorithm='RRT-Connect',iterations=iteration+1,collision_checks=self.checks,
                                     edge_resolution_deg=np.rad2deg(self.resolution),elapsed_s=time.monotonic()-began)
        raise UnsafePlan(f'RRT-Connect exhausted its bounded search ({self.checks} collision checks)')


def cartesian_joint_path(model,start_q,target_pose,*,step_mm=8,support_contact=False):
    start=model.fk(start_q);goal=pose_matrix(target_pose)
    if np.linalg.norm(Rotation.from_matrix(start[:3,:3]@goal[:3,:3].T).as_rotvec())>np.deg2rad(1):
        raise UnsafePlan('vertical approach requires a fixed starting orientation')
    distance=np.linalg.norm(goal[:3,3]-start[:3,3]);path=[np.array(start_q)]
    checker=RRTConnect(lambda q:model.valid(q,support_contact=support_contact),model.limits)
    for t in np.linspace(0,1,max(2,int(np.ceil(distance/(step_mm/1000)))+1))[1:]:
        pose=[*((start[:3,3]+t*(goal[:3,3]-start[:3,3]))*1000),*target_pose[3:]]
        q=model.ik(pose,path[-1],tries=3,support_contact=support_contact)
        if np.max(np.abs(q-path[-1]))>np.deg2rad(20) or not checker.edge(path[-1],q):raise UnsafePlan('vertical joint path discontinuity or collision')
        path.append(q)
    return path
