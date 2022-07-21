import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    # pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    # gazebo = launch.actions.ExecuteProcess(
    #         cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',
    #         'install/rmf_camera/share/rmf_camera/worlds/test.world'],
    #         output='screen'
    #     )
    pkg_ros_ign_gazebo = get_package_share_directory('ros_ign_gazebo')
    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py')),
        launch_arguments={'ign_args': '-r install/rmf_camera/share/rmf_camera/worlds/test_world.sdf'}.items(),
    )

    # Bridge
    bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/camera_sensor/pose/info@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
            ],
        output='screen',
    )
    bridge1 = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/camera_sensor/model/camera1/link/visual_link/sensor/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/world/camera_sensor/model/camera1/link/visual_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo'
            ],
        output='screen',
        remappings=[
            ('/world/camera_sensor/model/camera1/link/visual_link/sensor/camera/image', '/camera1/image_raw'),
            ('/world/camera_sensor/model/camera1/link/visual_link/sensor/camera/camera_info', '/camera1/camera_info'),
        ]
    )
    bridge2 = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/camera_sensor/model/camera2/link/visual_link/sensor/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/world/camera_sensor/model/camera2/link/visual_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo'
            ],
        output='screen',
        remappings=[
            ('/world/camera_sensor/model/camera2/link/visual_link/sensor/camera/image', '/camera2/image_raw'),
            ('/world/camera_sensor/model/camera2/link/visual_link/sensor/camera/camera_info', '/camera2/camera_info'),
        ]
    )
    return LaunchDescription([
        Node(
           package='rmf_camera',
           executable='YoloDetector'
        ),
        Node(
            package='image_proc',
            namespace='/camera1',
            executable='image_proc',
            remappings=[
                ('image', 'image_raw'),
            ]
        ),
        Node(
            package='image_proc',
            namespace='/camera2',
            executable='image_proc',
            remappings=[
                ('image', 'image_raw'),
            ]
        ),
        ign_gazebo,
        bridge,
        bridge1,
        bridge2,
    ])
