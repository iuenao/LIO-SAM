import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory('lio_sam')
    parameter_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(share_dir, 'config', 'R3LIVE.yaml'),
        description='Path to the pure LIO-SAM parameters file for R3LIVE runs.')

    use_sim_time_declare = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time when replaying R3LIVE rosbag data.')

    common_params = [parameter_file, {'use_sim_time': use_sim_time}]

    return LaunchDescription([
        params_declare,
        use_sim_time_declare,
        Node(
            package='lio_sam',
            executable='lio_sam_livoxConverter',
            name='lio_sam_livoxConverter',
            parameters=common_params,
            output='screen',
            remappings=[
                ('livox/mid360/lidar', '/livox/lidar'),
                ('livox/mid360/lidar_pc2', 'livox/lidar_pc2'),
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
            parameters=common_params,
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_imuPreintegration',
            name='lio_sam_imuPreintegration',
            parameters=common_params,
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_imageProjection',
            name='lio_sam_imageProjection',
            parameters=common_params,
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_featureExtraction',
            name='lio_sam_featureExtraction',
            parameters=common_params,
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_mapOptimization',
            name='lio_sam_mapOptimization',
            parameters=common_params,
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
