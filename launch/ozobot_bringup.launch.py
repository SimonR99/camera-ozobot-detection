"""Bring up the camera + robot bridge for Ozobot band detection.

Starts, in one shot, the equivalent of running these two commands:

    ros2 launch realsense2_camera rs_launch.py
    ros2 launch g1_ros2_bridge robot.launch.py

The RealSense node owns the camera and publishes /camera/color/image_raw, which
the detection scripts consume with `--ros-topic /camera/color/image_raw`. The
g1_ros2_bridge runs with its own camera disabled (its default) so the two do not
fight over the device.

Run (after sourcing ROS 2 and the unitree_g1_ros2 workspace):

    ros2 launch launch/ozobot_bringup.launch.py

Pass-through arguments:
    pointcloud_enable:=false        # skip the depth point cloud
    g1_interface:=eth0              # Unitree DDS network interface (else $G1_INTERFACE)
    enable_camera:=false            # keep the bridge's own camera off (default)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    realsense_share = get_package_share_directory("realsense2_camera")
    g1_share = get_package_share_directory("g1_ros2_bridge")

    pointcloud_enable = LaunchConfiguration("pointcloud_enable")
    g1_interface = LaunchConfiguration("g1_interface")
    enable_camera = LaunchConfiguration("enable_camera")

    args = [
        DeclareLaunchArgument(
            "pointcloud_enable",
            default_value="true",
            description="Publish the RealSense depth/color point cloud",
        ),
        DeclareLaunchArgument(
            "g1_interface",
            default_value="",
            description="Unitree DDS network interface (falls back to $G1_INTERFACE)",
        ),
        DeclareLaunchArgument(
            "enable_camera",
            default_value="false",
            description="Let g1_ros2_bridge launch its own RealSense node "
            "(keep false; this file already starts the camera)",
        ),
    ]

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, "launch", "rs_launch.py")
        ),
        launch_arguments={"pointcloud.enable": pointcloud_enable}.items(),
    )

    g1_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(g1_share, "launch", "robot.launch.py")
        ),
        launch_arguments={
            "enable_camera": enable_camera,
            "interface": g1_interface,
        }.items(),
    )

    return LaunchDescription([*args, realsense, g1_bridge])
