#!/usr/bin/env python3
"""
knock_sim_cli.py
Headless (or GUI) micro-sim to knock a target cube and anything above it,
then return final poses for all cubes.

Request JSON (stdin or --json file):
{
  "edge": 0.05,
  "direction": [1.0, 0.0],
  "friction": 0.6,
  "restitution": 0.0,
  "mass": 0.2,
  "gui": false,
  "hold": false,                   # GUI-only: keep window open (manual debug)
  "mode": "force",                 # "force"|"torque"|"velocity"
  "knock_id": "g2",
  "cubes": [
    {"id":"r1","position":[x,y,z],"orientation":[qx,qy,qz,qw]},
    ...
  ]
}

Response JSON (stdout):
{"poses":[{"id":"...","position":[...],"orientation":[...]}, ...]}
"""

import os
import sys
import json
import time
import argparse
import math
from contextlib import contextmanager


# ---------- silence C-level stdout so PyBullet can't pollute our JSON ----------
@contextmanager
def silence_c_stdio():
    """Temporarily silence C-level stdout (fd=1) and stderr (fd=2)."""
    sys.stdout.flush()
    #sys.stderr.flush()
    saved_out = os.dup(1)
    #saved_err = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)   # stdout -> /dev/null
        #os.dup2(devnull, 2)   # stderr -> /dev/null
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_out, 1)
        #os.dup2(saved_err, 2)
        os.close(saved_out)
        #os.close(saved_err)

# ---------- color-from-name (GUI only; physics unaffected) ----------
def _rgba_from_id(name: str):
    s = name.lower()
    if s.startswith("red") or s.startswith("r"):
        return [0.85, 0.10, 0.10, 1.0]
    if s.startswith("green") or s.startswith("g"):
        return [0.20, 0.80, 0.20, 1.0]
    if s.startswith("blue") or s.startswith("b"):
        return [0.20, 0.40, 0.95, 1.0]
    return [0.80, 0.80, 0.80, 1.0]  # default gray

# ---------- compute displaced set from full stack ----------
def _compute_displaced_ids(cubes_in, knock_id, edge):
    """
    Return [knock_id] + all ids with higher Z in the same XY column.
    XY 'same column' is checked with a tolerance relative to cube size.
    """
    # tolerance: generous but safe; if your stacks are exactly aligned, 0.02 is fine too
    xy_tol = max(0.02, 0.4 * float(edge))
    # index by id
    by_id = {c["id"]: c for c in cubes_in}
    if knock_id not in by_id:
        raise ValueError(f"knock_id '{knock_id}' not found in 'cubes'")

    tp = by_id[knock_id]["position"]
    tx, ty, tz = float(tp[0]), float(tp[1]), float(tp[2])

    displaced = [knock_id]
    for c in cubes_in:
        if c["id"] == knock_id:
            continue
        x, y, z = float(c["position"][0]), float(c["position"][1]), float(c["position"][2])
        same_col = math.hypot(x - tx, y - ty) <= xy_tol
        if same_col and z > tz + 1e-9:
            displaced.append(c["id"])
    return displaced

# ---------- core simulation ----------
def simulate_knock(req: dict) -> dict:
    edge        = float(req.get("edge", 0.05))
    half        = edge / 2.0
    friction    = float(req.get("friction", 0.6))
    restitution = float(req.get("restitution", 0.0))
    mass        = float(req.get("mass", 0.2))
    gui         = bool(req.get("gui", False))
    hold        = bool(req.get("hold", False))
    mode        = str(req.get("mode", "force"))
    knock_id   = req.get("knock_id", "")
    cubes_in    = req["cubes"]
    dir_in      = req.get("direction", [1.0, 0.0])

    if not knock_id:
        raise ValueError("Missing 'knock_id' in request")

    # normalize direction to XY unit vector
    dx, dy = float(dir_in[0]), float(dir_in[1])
    n = math.hypot(dx, dy) or 1.0
    direction = [dx / n, dy / n, 0.0]

    # AUTOMATIC displaced subset from full stack (target + above in same column)
    displaced_ids = _compute_displaced_ids(cubes_in, knock_id, edge)

    with silence_c_stdio():
        import pybullet as pb
        import numpy as np
        cid = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setGravity(0, 0, -9.81)
        dt = 1.0 / 240.0
        pb.setTimeStep(dt)
        pb.setRealTimeSimulation(0)

        if gui:
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.25, cameraYaw=25, cameraPitch=-40,
                cameraTargetPosition=[0.5, 0.0, 0.06]
            )
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)

        # table
        floor_z = 0.0
        table_half = [2.5, 2.5, 0.01]
        shape_tab = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=table_half)
        vis_tab   = pb.createVisualShape(pb.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.7,0.7,0.7,1.0])
        table = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape_tab,
            baseVisualShapeIndex=vis_tab,
            basePosition=[0, 0, floor_z - table_half[2]],
        )
        pb.changeDynamics(table, -1, lateralFriction=friction, restitution=restitution)

        # cubes
        shape_cube = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half, half, half])
        id2body = {}
        ids_in_order = []
        for c in cubes_in:
            cid_ = c["id"]
            pos  = c["position"]
            orn  = c["orientation"]
            ids_in_order.append(cid_)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half,half,half],
                                       rgbaColor=_rgba_from_id(cid_))
            bid = pb.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=shape_cube,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                baseOrientation=orn,
            )
            pb.changeDynamics(bid, -1,
                              lateralFriction=friction,
                              restitution=restitution,
                              linearDamping=0.001,
                              angularDamping=0.001)
            id2body[cid_] = bid

        # small settle
        for _ in range(int(0.5 / dt)):
            pb.stepSimulation()
            if gui:
                time.sleep(dt)

        # knock axis = ẑ × dir = [-dy, dx, 0], normalized
        axis = [-direction[1], direction[0], 0.0]
        na = math.hypot(axis[0], axis[1]) or 1.0
        axis = [axis[0]/na, axis[1]/na, 0.0]

        # push burst
        if mode == "velocity":
            omega = 3.0
            v_lin = 0.05
            for tid in displaced_ids:
                if tid not in id2body:
                    continue
                pb.resetBaseVelocity(
                    id2body[tid],
                    linearVelocity=[direction[0]*v_lin, direction[1]*v_lin, 0.0],
                    angularVelocity=[axis[0]*omega, axis[1]*omega, 0.0]
                )
            push_frames = 1

        elif mode == "torque":
            tau = [axis[0]*mass*9.81*edge*0.6,
                   axis[1]*mass*9.81*edge*0.6,
                   0.0]
            push_frames = 12

        else:  # "force"
            #F_push = [direction[0]*1.6*mass*9.81, direction[1]*1.6*mass*9.81, 0.0]
            F_push = [direction[0]*5.0*mass*9.81, direction[1]*5.0*mass*9.81, 0.0]
            contact_points = {}
            for tid in displaced_ids:
                if tid not in id2body:
                    continue
                p, q = pb.getBasePositionAndOrientation(id2body[tid])
                R = np.array(pb.getMatrixFromQuaternion(q)).reshape(3,3)
                r_local = np.array([direction[0]*half, direction[1]*half, half])  # COM → top-edge
                contactW = (np.array(p) + R @ r_local).tolist()
                contact_points[tid] = contactW
            push_frames = 12

        for _ in range(push_frames):
            if mode == "torque":
                for tid in displaced_ids:
                    if tid not in id2body:
                        continue
                    pb.applyExternalTorque(id2body[tid], -1, torqueObj=tau, flags=pb.WORLD_FRAME)
            elif mode == "force":
                for tid in displaced_ids:
                    if tid not in id2body:
                        continue
                    pb.applyExternalForce(id2body[tid], -1, forceObj=F_push,
                                          posObj=contact_points[tid], flags=pb.WORLD_FRAME)
            pb.stepSimulation()
            if gui:
                time.sleep(dt)

        # settle
        def _stable(eps_v=0.02, eps_w=0.05):
            for bid in id2body.values():
                lin, ang = pb.getBaseVelocity(bid)
                if (lin[0]*lin[0] + lin[1]*lin[1] + lin[2]*lin[2]) ** 0.5 > eps_v:
                    return False
                if (ang[0]*ang[0] + ang[1]*ang[1] + ang[2]*ang[2]) ** 0.5 > eps_w:
                    return False
            return True

        quiet = 0
        for _ in range(int(4.0 / dt)):
            pb.stepSimulation()
            if gui:
                time.sleep(dt)
            if _stable():
                quiet += 1
                if quiet > int(0.15 / dt):
                    break
            else:
                quiet = 0

        # collect poses (same order as input)
        out_poses = []
        for cid_ in ids_in_order:
            pos, orn = pb.getBasePositionAndOrientation(id2body[cid_])
            out_poses.append({
                "id": cid_,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "orientation": [float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])],
            })

        if gui and hold:
            while pb.isConnected():
                time.sleep(0.05)

        pb.disconnect()

    return {"poses": out_poses}

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Path to request JSON (else read stdin)")
    parser.add_argument("--out",  help="Path to write response JSON (else stdout)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print the received request to STDERR")
    args = parser.parse_args()

    if args.json:
        with open(args.json, "r") as f:
            req = json.load(f)
    else:
        req = json.load(sys.stdin)

    print(args)
    if args.verbose:
        print("[knock] request:", json.dumps(req, indent=2, sort_keys=True),
              file=sys.stderr, flush=True)

    res = simulate_knock(req)
    print(res, file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)
    else:
        json.dump(res, sys.stdout)
        sys.stdout.flush()

if __name__ == "__main__":
    main()

