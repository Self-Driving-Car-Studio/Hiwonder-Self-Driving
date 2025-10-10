import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    start = LaunchConfiguration('start', default='true')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    only_line_follow = LaunchConfiguration('only_line_follow', default='false')
    only_line_follow_arg = DeclareLaunchArgument('only_line_follow', default_value=only_line_follow)
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        controller_package_path = get_package_share_directory('controller')
        package_share_directory = get_package_share_directory('yolov5_ros2')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        package_share_directory = '/home/ubuntu/ros2_ws/src/yolov5_ros2'

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        output='screen',
    )
    yolov5_node = Node(
        package='yolov5_ros2',
        executable='yolo_detect',
        output='screen',
        parameters=[{'classes': ['go', 'right', 'park', 'red', 'green', 'crosswalk', 'yellow', 'right_sign']},
            {"device": "cpu",
            "model": "traffic_v4_s",
            "image_topic": "/ascamera/camera_publisher/rgb0/image",
            "camera_info_topic": "/camera/camera_info",
            "camera_info_file": f"{package_share_directory}/config/camera_info.yaml",
            "pub_result_img": True}]
    )

    self_driving_node = Node(
        package='example',
        executable='self_driving',
        output='screen',
        parameters=[{'start': start}, {'only_line_follow': only_line_follow}],
    )

    return [start_arg,
            only_line_follow_arg,
            depth_camera_launch,
            controller_launch,
            yolov5_node, 
            self_driving_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
