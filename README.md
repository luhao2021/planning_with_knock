
# TAMP(Task and Motion Planning) Knocking Task

This repository includes software packages that enable the task and motion planning strategies with knocking operations, and the strategies are demonstrated with Franka Emika Panda in MoveIt2 and the IsaacSim simulator

## Overview

This package is tested with Ubuntu 22.04 Jammy, ROS 2 `humble` and IsaacSim 5.0.0. The sturcture of this packages are described as below:

- [**problems**](./problems) – Problem files, loaded and executed by the knocking demo
- [**blocksworld_scene**](./src/blocksworld_scene) – Lauch files for the knock environment, python script of the demo
- [**blocksworld_executor**](./src/blockworld_executor) – Executor of the TAMP plan actions

## Instructions

### Dependencies

These are the primary dependencies required to use this project.

- ROS 2 [Humble](https://docs.ros.org/en/humble/Installation.html)
- Moveit 2 [Humble](https://moveit.picknik.ai/humble/index.html)
- IsaacSim [5.0.0](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html)

This package assume you have ROS 2 Humble, MoveIt 2 Humble, and IsaacSim installed. Additional dependencies can be pulled via [vcstool](https://wiki.ros.org/vcstool) ([dependencies.repos](./src/dependencies.repos)). Please see below for details of building process.

### Building

Clone this repository, import dependencies, install dependencies and build with [colcon](https://colcon.readthedocs.io).

```bash
# Build
cd tamp_knock
source $(MOVEIT2_OVERLAY_DIRECTORY)/install/setup.bash
colcon build --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
```

### Run the knocking demo

Please follow below steps to run the demo

```bash
# Terminal 1
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib/
export ISAACSIM_PATH="/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
source /home/luhao/Workspace/planning/isaacsim-ros-ws/build_ws/humble/isaac_sim_moveit_ws/install/setup.bash
$ISAACSIM_PYTHON_EXE src/isaac_moveit/scripts/scene_isaacsim.py --yaml my_problem_move.yaml

# Terminal 2
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source ../moveit_ws/install/setup.bash
source ./install/local_setup.bash
ros2 launch isaac_moveit isaac_moveit.launch.py

# Terminal 3
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source ../moveit_ws/install/setup.bash
source install/local_setup.bash
ros2 launch blocksworld_executor smash_demo.launch.py problem:=my_problem_move.yaml
```

### Run the UFactory XARM7 Real-robot Demo
```bash
# Terminal 1
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib/
export ISAACSIM_PATH="/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
source /home/luhao/Workspace/planning/isaacsim-ros-ws/build_ws/humble/isaac_sim_moveit_ws/install/setup.bash
$ISAACSIM_PYTHON_EXE src/isaac_moveit/scripts/scene_isaacsim.py --yaml problem_1x6_video2.yaml

# Terminal 2
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source ../moveit_ws/install/setup.bash
source ./install/local_setup.bash
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py [add_gripper:=true]

# Terminal 3
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source ../moveit_ws/install/setup.bash
source ./install/local_setup.bash
ros2 launch blocksworld_executor smash_demo_xarm7.launch.py problem:=problem_1x6_video2.yaml

Then file "/tmp/mtc_export.yaml" is generated which include the plan trajectory sequence, and then copy to ufactory machine:
scp /tmp/mtc_export.yaml ufactory@192.168.1.12:~/

Run below on ufactory machine to play the trajectory (might need to update the gripper open position):
# Terminal 1 (on ufactory machine)
source /opt/ros/noetic/setup.bash
source ~/workspaces/gazebo_catkin_ws/devel/setup.bash
source ~/workspaces/ufactory_ws/devel/setup.bash
source ~/workspaces/sensing_ws/devel/setup.bash
source ~/workspaces/sensing_ws/devel_isolated/setup.bash

roslaunch xarm7_gripper_moveit_config realMove_exec.launch robot_ip:=192.168.1.231 [velocity_control:=false] [report_type:=normal]
python3 replay_mtc_path.py ./mtc_export5.yaml --time-scale 10
```

### Run the benchmark script
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
./run_bench.py
# run below to show the statistical results
python3 summarize.py runs/20250915-2235/k1x9/scenario_summary.json
```

### Run the environment generation script
```bash
python3 scripts/gen_env.py  --m 2 --n 3  --size 0.05 --gap_s 0.01  --gap_d 0.03 --center_x 0.45 --center_y 0.00 --seed 42 --output blocks.yaml
```

### Debug

1. Update the pose of the object in the scene:
```
ros2 topic pub /planning_scene moveit_msgs/msg/PlanningScene "{
  is_diff: true,
  world: {
    collision_objects: [{
      id: 'wood_cube_red_visual',
      header: {
        frame_id: 'world'
      },
      pose: {
        position: { x: 0.45, y: -0.3, z: 0.025 },
        orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
      },
      operation: [0x03]
    }]
  }
}"
```
