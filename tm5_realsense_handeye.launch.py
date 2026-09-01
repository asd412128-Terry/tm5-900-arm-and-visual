"""
TM5-900 + RealSense D435i Eye-in-Hand 手眼標定 launch 檔（正確版）
====================================================================
直接 include easy_handeye2 官方提供的 calibrate.launch.py。

跑之前，除了這個 launch 檔，還需要「分開」啟動：
  1. TM5-900 driver + MoveIt2
  2. RealSense driver
  3. checkerboard_pose_publisher.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    easy_handeye2_calibrate_launch = PathJoinSubstitution(
        [FindPackageShare('easy_handeye2'), 'launch', 'calibrate.launch.py']
    )

    calibrate_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(easy_handeye2_calibrate_launch),
        launch_arguments={
            'calibration_type': 'eye_in_hand',
            'name': 'tm5_realsense_handeye',
            'robot_base_frame': 'base',
            'robot_effector_frame': 'link_6',
            'tracking_base_frame': 'camera_color_optical_frame',
            'tracking_marker_frame': 'checkerboard_frame',
            'freehand_robot_movement': 'true',
        }.items(),
    )

    return LaunchDescription([calibrate_include])