import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    gazebo = launch.actions.ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',
            'install/rmf_camera/share/rmf_camera/worlds/test.world'],
            output='screen'
        )
    return LaunchDescription([
        # Node(
        #     package='rmf_camera',
        #     executable='YoloDetector'
        # ),
        Node(
            package='rmf_obstacle_ros2',
            executable='obstacle_manager_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"detector_plugin": "rmf_human_detector::HumanDetector"}
            ]
        ),
        Node(
            package='image_proc',
            namespace='/camera',
            executable='image_proc',
            remappings=[
                ('image', '/camera/image_raw'),
            ]
        ),
        gazebo
    ])
