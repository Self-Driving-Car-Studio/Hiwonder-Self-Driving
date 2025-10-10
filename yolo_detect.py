from math import frexp
from traceback import print_tb
from torch import imag
from yolov5 import YOLOv5
import rclpy
import yolov5_ros2.fps as fps
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
#from sdk import common

from yolov5_ros2.cv_tool import px2xy
import os
from interfaces.msg import ObjectInfo, ObjectsInfo

from std_srvs.srv import Trigger

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))

        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /ascamera/camera_publisher/rgb0/image"))

        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))

        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Load the model.
        model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".onnx"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # Create publishers.
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # Create an image subscriber with the updated topic.
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        # Image format conversion (using cv_bridge).
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

        ################################
        # [추가] 탐지 빈도 조절을 위한 변수
        self.frame_counter = 0
        self.detect_interval = 5  # 5프레임당 1번씩만 탐지
        self.last_objects_msg = ObjectsInfo() # 마지막 탐지 결과를 저장할 변수

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolov5 detect")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolov5 detect")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    ### 수정 ###
    def image_callback(self, msg: Image):

        self.frame_counter += 1
        
        # [수정] 정해진 간격(detect_interval)마다 탐지 수행
        if self.frame_counter % self.detect_interval == 0:
            image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            detect_result = self.yolov5.predict(image)

            # 결과를 파싱하고 저장
            predictions = detect_result.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            objects_info = []
            h, w = image.shape[:2]
            
            for index in range(len(categories)):
                name = detect_result.names[int(categories[index])]
                x1, y1, x2, y2 = map(int, boxes[index])

                object_info = ObjectInfo()
                object_info.class_name = name
                object_info.box = [x1, y1, x2, y2]
                object_info.score = round(float(scores[index]), 2)
                objects_info.append(object_info)

            # [수정] 이번 프레임의 탐지 결과를 self.last_objects_msg에 저장
            self.last_objects_msg.objects = objects_info
            
            # (시각화 로직은 그대로 유지)
            if self.show_result or self.pub_result_img:
                for obj in objects_info:
                    cv2.rectangle(image, (obj.box[0], obj.box[1]), (obj.box[2], obj.box[3]), (0, 255, 0), 2)
                    cv2.putText(image, f"{obj.class_name}:{obj.score:.2f}", (obj.box[0], obj.box[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
                if self.show_result:
                    self.fps.update()
                    image = self.fps.show_fps(image)
                    cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB_BGR))
                    cv2.waitKey(1)

                if self.pub_result_img:
                    result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
                    result_img_msg.header = msg.header
                    self.result_img_pub.publish(result_img_msg)

        # [수정] 마지막 탐지 결과를 매 프레임 발행하여 self_driving 노드가 항상 정보를 받도록 함
        if self.last_objects_msg.objects:
            self.object_pub.publish(self.last_objects_msg)
  

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()

