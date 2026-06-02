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
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz_m3dgr.rviz')
    ground_yaw = LaunchConfiguration('ground_yaw')
    ground_pitch = LaunchConfiguration('ground_pitch')
    ground_roll = LaunchConfiguration('ground_roll')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(share_dir, 'config', 'M3DGR.yaml'),
        description='Path to the pure LIO-SAM parameters file for M3DGR runs.')

    use_sim_time_declare = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time when replaying M3DGR rosbag data.')

    ground_yaw_declare = DeclareLaunchArgument(
        'ground_yaw',
        default_value='-1.4724742460100058',
        description='Display-only yaw for m3dgr_ground -> map.')

    ground_pitch_declare = DeclareLaunchArgument(
        'ground_pitch',
        default_value='-0.04095004393683892',
        description='Display-only pitch for m3dgr_ground -> map.')

    ground_roll_declare = DeclareLaunchArgument(
        'ground_roll',
        default_value='-1.0288150588549427',
        description='Display-only roll for m3dgr_ground -> map.')

    common_params = [parameter_file, {'use_sim_time': use_sim_time}]

    return LaunchDescription([
        params_declare,
        use_sim_time_declare,
        ground_yaw_declare,
        ground_pitch_declare,
        ground_roll_declare,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '0.0', '0.0', '0.0',
                ground_yaw, ground_pitch, ground_roll,
                'm3dgr_ground', 'map'
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_livoxConverter',
            name='lio_sam_livoxConverter',
            parameters=common_params,
            output='screen'
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
