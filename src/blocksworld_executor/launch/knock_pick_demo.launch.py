from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    # Declare use_sim_time argument
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )


    knock_pick_executor = Node(package='blocksworld_executor',
        executable='knock_pick_executor',
        output="screen",
        emulate_tty=True,
        #prefix='xterm -fa "Monospace" -fs 12 -geometry 100x55 -T "Knock-Pick Demo Terminal" -e',
        parameters=[
            {'problem': LaunchConfiguration('problem')},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            moveit_config.to_dict()
        ]
    )

    return LaunchDescription(
        [
            ros2_control_hardware_type,
            use_sim_time,  # Declare use_sim_time argument here
            knock_pick_executor,
        ]
    )
