#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
# from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState, RGBStates, RGBState # for RGB LED control import: RGBStates, RGBState #jr

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True

        self.pid = pid.PID(0.2, 0.0, 0.1) # D 제어부분 0.1 로 증가
        self.turn_pid = pid.PID(0.2, 0.0, 0.4) # 일반pid 와 turn 전용 pid 구분, turn 의 D부분을 올림
  
        self.param_init() # 모든 상태변수 초기화

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk','yellow','right_sign']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # [추가] LED 상태 추적 및 깜빡임 로직을 위한 변수
        self.led1_color = (-1, -1, -1) # LED의 현재 색상 저장 (중복 전송 방지)
        self.led2_color = (-1, -1, -1)  # LED의 현재 색상 저장 (중복 전송 방지)
        self.last_blink_time = 0

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.rgb_pub = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 1) # for RGB LED control #jr


        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

        # 노드 시작 시 LED를 초기 상태(꺼짐)로 설정
        self.set_led_color(1, 0, 0, 0)
        self.set_led_color(2, 0, 0, 0)

    
    # [추가] LED 제어 함수 - 중복 전송 방지 로직 추가
    def set_led_color(self, led_id, r, g, b):
        target_color = (r, g, b)
        # 현재 색상과 목표 색상이 다를 때만 메시지를 보내 효율성 증대
        if led_id == 1 and self.led1_color == target_color:
            return
        if led_id == 2 and self.led2_color == target_color:
            return

        msg = RGBStates()
        state = RGBState()
        state.index = led_id
        state.red = r
        state.green = g
        state.blue = b
        msg.states.append(state)
        self.rgb_pub.publish(msg)
        
        # 현재 색상 상태 업데이트
        if led_id == 1:
            self.led1_color = target_color
        else:
            self.led2_color = target_color


    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        if 1:#self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        #self.park_action()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.4 # normal driving speed # 횡단보도 인식을 위해 줄이는중..
        self.slow_down_speed = 0.2  # slowing down speed
        self.cornering_speed = 0.2 # 코너링 스피드 ( 미리 진입하기 전부터 작동 )

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []
        # [추가] "주차 완료" 상태를 초기화합니다.
        self.parking_completed = False

        # 추가 - crosswalk 를 인식하고 정지하기 위한 변수선언 
        self.target_crosswalk = None # 정지 목표가 되는 횡단보도 객체 정보
        self.passed_first_crosswalk = False # 첫 번째 횡단보도를 통과했는지 여부
        self.stop_for_crosswalk_start_time = 0 # 횡단보도 앞 정지 시작 시간
        self.is_passing_intersection = False  # 현재 교차로를 통과/진행 중인지 여부
        self.crosswalk_disappear_counter = 0  # 횡단보도가 안 보이기 시작한 프레임 수

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            camera = 'depth_cam' #self.get_parameter('depth_camera_name').value
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        ### 시간 측정 시작 ###
        reception_time = time.time()
        ### 시간 측정 종료 ###

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put((rgb_image, reception_time))
    
    # parking processing
    def park_action(self):
        if self.machine_type == 'MentorPi_Mecanum':
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38/0.2)
        elif self.machine_type == 'MentorPi_Acker':
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65/0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())
        # [추가] 모든 주차 동작이 끝나면 "주차 완료" 상태 플래그를 True로 설정합니다.
        self.parking_completed = True
        self.get_logger().info('주차 완료! LED를 소등합니다.')

    def main(self):
        while self.is_running:
            time_start = time.time() # 루프 시작 시간
            
            items_in_queue = [] 
            # 1. 큐가 빌 때까지 모든 항목을 꺼내서 임시 리스트에 담습니다. (줄 세워 내보내기)
            while not self.image_queue.empty():
                try:
                    items_in_queue.append(self.image_queue.get_nowait())
                except queue.Empty:
                    continue
            
            # 2. 임시 리스트가 비어있으면 처리할 게 없으므로 건너뜁니다.
            if not items_in_queue:
                time.sleep(0.001)
                continue

            image, reception_time = items_in_queue[-1]

            ### 시간 측정 시작 ###
            # 큐 대기 시간 측정
            queue_wait_time = (time_start - reception_time) * 1000 # ms
            t1 = time.time()
            ### 시간 측정 종료 ###

            result_image = image.copy()
            if self.start:
                # ### [추가] LED 상태 제어 로직 ###
                 # [수정] LED 상태 제어 로직 (우선순위 적용)
                # 우선순위 0: 주차 완료
                if self.parking_completed:
                    self.set_led_color(1, 0, 0, 0) # 끄기
                    self.set_led_color(2, 0, 0, 0) # 끄기
                # 우선순위 1: 정지 상태 (stop 플래그)
                elif self.stop:
                    self.set_led_color(1, 255, 0, 0) # 빨간색
                    self.set_led_color(2, 255, 0, 0) # 빨간색
                # 우선순위 2: 우회전 중 (start_turn 플래그)
                elif self.start_turn:
                    self.set_led_color(1, 0, 255, 0) # 1번 LED는 초록색 유지
                    if time.time() - self.last_blink_time > 0.25:
                        self.last_blink_time = time.time()
                        if self.led2_color == (0, 0, 0):
                            self.set_led_color(2, 255, 255, 0) # 노란색 켜기
                        else:
                            self.set_led_color(2, 0, 0, 0) # 끄기
                # 우선순위 3: 일반 주행 상태
                else:
                    self.set_led_color(1, 0, 255, 0) # 초록색
                    self.set_led_color(2, 0, 255, 0) # 초록색
                # #################################
                
                
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)
                t2 = time.time()

                twist = Twist()

                # 아래코드는 위와 겹치므로 주석처리 / 확정나면 삭제

                # if detecting the zebra crossing, start to slow down
                # 슬로우 다운 모드 필요없어서 주석처리함

                # self.get_logger().info('\033[1;33m%s\033[0m' % self.crosswalk_distance)
                # if 200 < self.crosswalk_distance and not self.start_slow_down:  #바꿈 jr 원래값 70 # The robot starts to slow down only when it is close enough to the zebra crossing
                #     self.count_crosswalk += 1
                #     if self.count_crosswalk == 3:  # judge multiple times to prevent false detection
                #         self.count_crosswalk = 0
                #         self.start_slow_down = True  # sign for slowing down
                #         self.count_slow_down = time.time()  # fixing time for slowing down
                # else:  # need to detect continuously, otherwise reset
                #     self.count_crosswalk = 0

                ### 횡단보도 로직 추가

                                # 횡단보도 정지 관련 로직

                t3_start = time.time()
                
                crosswalk_stop_triggered = False # 차선유지 로직에 쓰임

                # 1. 주요 타겟 횡단보도가 있고, 아직 정지 동작을 시작하지 않았을 때
                if self.target_crosswalk is not None and self.stop_for_crosswalk_start_time == 0 :
                    box = self.target_crosswalk.box
                    area = abs(box[2] - box[0]) * abs(box[3]-box[1])
                    self.get_logger().info(f"타겟 횡단보도 확인, 면적: {area:.0f}")

                    # 1-1. 면적이 정지 기준을 충족하고, 아직 정지 타이머가 시작되지 않았다면
                    if self.target_crosswalk.class_name == 'crosswalk' and 25000 > area > 9000 and self.stop_for_crosswalk_start_time == 0:
                        self.get_logger().info("횡단보도 근접, 정지중..")
                        self.mecanum_pub.publish(Twist()) # 정지
                        self.stop_for_crosswalk_start_time = time.time() # 현재 정지한 시간
                        crosswalk_stop_triggered = True  # 횡단보도 때문에 멈췄다고 True

                t3_end = time.time()
                
                # 2. 횡단보도 앞ㅇ있는 동안 로직
                # 2. 횡단보도 앞에 정지해 있는 동안의 로직
                if self.stop_for_crosswalk_start_time > 0: # 현재 멈춰있는 상태인가?
                    crosswalk_stop_triggered = True # 정지 중이므로 차선유지 로직 비활성화
                    can_go = False
                    
                    if self.traffic_signs_status is not None:
                        if self.traffic_signs_status.class_name == 'green':
                            self.get_logger().info("Green light detected. Go!")
                            can_go = True
                        elif self.traffic_signs_status.class_name == 'red':
                            self.get_logger().info("Red light detected. Waiting...")
                            self.mecanum_pub.publish(Twist()) # 정지유지
                            # crosswalk_stop_triggered = True
                    else: # 신호등이 없을 때
                        if time.time() - self.stop_for_crosswalk_start_time > 1.0: # 1초 대기
                            self.get_logger().info("No traffic light. Go after 1 sec.")
                            can_go = True
                        else:
                            self.mecanum_pub.publish(Twist())
                            # crosswalk_stop_triggered = True
                            
                    if can_go:
                        self.is_passing_intersection = True # '통과 중' 상태로 전환
                        self.stop_for_crosswalk_start_time = 0

                # 3. 교차로 통과 완료 및 상태 초기화 로직
                if self.is_passing_intersection:
                    current_crosswalks_count = sum(1 for obj in self.objects_info if obj.class_name == 'crosswalk')
                    if current_crosswalks_count < 2: # 횡단보도가 2개 미만으로 보이면
                        self.crosswalk_disappear_counter += 1
                    else:
                        self.crosswalk_disappear_counter = 0
                    
                    if self.crosswalk_disappear_counter > 30: # 약 1초간 횡단보도가 안 보이면
                        self.get_logger().info("Intersection cleared. Ready for the next one.")
                        self.is_passing_intersection = False
                        self.crosswalk_disappear_counter = 0




                # 3. 정지 동작이 시작됐고 1초가 지났을 때
                # if self.stop_for_crosswalk_start_time > 0 and (time.time() - self.stop_for_crosswalk_start_time > 1.0):

                #     # 1초가 지났고, 신호등 관련
                #     if self.traffic_signs_status is not None: # 신호등이 감지가 됐다면
                #         area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                #         if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                #             self.mecanum_pub.publish(Twist()) #정지
                #             self.stop = True
                #         elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                #             twist.linear.x = self.normal_speed # 노말 스피드로 통과
                #             self.stop = False

                #             self.get_logger().info("1초 경과, 주행 모드")
                #             self.passed_first_crosswalk = True # 첫 번째 횡단보도 통과 처리
                #             self.stop_for_crosswalk_start_time = 0 # 정지 시간 초기화

                            
                #             # 최종 명령전달
                #             self.mecanum_pub.publish(Twist()) # (전진)
                            
                #             # 첫번째 횡단보도를 지났다면 초기화
                            
                #             self.passed_first_crosswalk = False
                            

                #     # 정지 1초가 지나고 신호등이 없다면        
                #     else:
                #         self.get_logger().info("1초 경과, 주행 모드")
                #         self.passed_first_crosswalk = True # 첫 번째 횡단보도 통과 처리
                #         self.stop_for_crosswalk_start_time = 0 # 정지 시간 초기화
                #         twist.linear.x = self.normal_speed  # go straight with normal speed

                #         # 최종 명령전달
                #         self.mecanum_pub.publish(twist) # (전진)
                #         # 기존에 publish(Twist()) 였으나,
                #         # 이는 속도 0을 퍼블리시 하는 객체임.


                #         # 첫번째 횡단보도를 지났다면 초기화
                #         self.passed_first_crosswalk = True






                # # deceleration processing
                # # 슬로우 다운이라면 -> 횡단보도가 있다면
                # if self.start_slow_down:

                #     if self.traffic_signs_status is not None: # 신호등이 감지가 됐다면
                #         area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                #         if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                #             self.mecanum_pub.publish(Twist()) #정지
                #             self.stop = True
                #         elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                #             twist.linear.x = self.slow_down_speed # 슬로우다운 스피드로 통과
                #             self.stop = False

                #     # 신호등이 없다면
                #     if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                #         twist.linear.x = self.slow_down_speed
                #         if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                #             self.start_slow_down = False

                # # 슬로우 다운이 아니라면 -> 횡단보도가 없다면 ( 일반주행 )
                # else:
                #     twist.linear.x = self.normal_speed  # go straight with normal speed

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # 거리는 안쓰니까 주석
                # if 0 < self.park_x and 135 < self.crosswalk_distance:
                #     twist.linear.x = self.slow_down_speed
                #     if not self.start_park and 180 < self.crosswalk_distance:  # When the robot is close enough to the crosswalk, it will start parking
                #         self.count_park += 1
                #         if self.count_park >= 15:
                #             self.mecanum_pub.publish(Twist())
                #             self.start_park = True
                #             self.stop = True
                #             threading.Thread(target=self.park_action).start()
                #     else:
                #         self.count_park = 0

                ### 시간 측정 시작 ###
                # 2. 차선 인식 및 제어 로직 시간 측정
                t3_start = time.time()
                ### 시간 측정 종료 ###

                # 횡단보도 때문에 안 멈췄다면 ( False 가 아니라면 / 기본값 False)
                if not crosswalk_stop_triggered: 


                    # line following processing
                    result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
                    if lane_x >= 0 and not self.stop:
                        

                        if lane_x > 180: # 1순위 코너감지, 턴 시작
                            
                            # 우회전으로 판단하고 mode 전환
                            self.lane_detect.set_mode("close_up")

                            self.count_turn += 1
                            if self.count_turn > 5 and not self.start_turn:
                                self.start_turn = True
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            if self.machine_type != 'MentorPi_Acker':
                                twist.linear.x = self.cornering_speed # 코너링 속도
                                twist.angular.z = -1.2   # turning speed
                            else:
                                twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145

                        elif lane_x > 150: # 코너링 미리 감지하고 준비 ( 속도감소 ) - 2순위
                            twist.linear.x = self.cornering_speed
                            # 감속 중에도 PID 제어는 유지
                            self.pid.SetPoint = 130 
                            self.pid.update(lane_x)
                            twist.angular.z = -common.set_range(self.pid.output, -0.3, 0.3)

                        else:  # use PID algorithm to correct turns on a straight road

                            # 직진으로 판단하고 mode전환
                            # 항상 턴 돌고 횡단보도를 인식못하는 문제 발생 잠깐 슬로우 모드 넣어야할듯
                            self.lane_detect.set_mode("look_ahead")
                            twist.linear.x = self.normal_speed # 일반 속도
                            


                            self.count_turn = 0
                            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                                self.start_turn = False
                            if not self.start_turn:
                                self.pid.SetPoint = 130  # the coordinate of the line while the robot is in the middle of the lane
                                self.pid.update(lane_x)
                                if self.machine_type != 'MentorPi_Acker':
                                    twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                            else:
                                if self.machine_type == 'MentorPi_Acker':
                                    twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                        self.mecanum_pub.publish(twist)
                    else:
                        # 차선인식 못한 경우
                        self.lane_detect.set_mode("close_up")

                        # 느린 명령주면서 차선 찾기
                        recovery_twist = Twist()
                        recovery_twist.linear.x = 0.1  # 느린 속도로
                        recovery_twist.angular.z = 0.4   # 왼쪽으로 살짝 회전
                        self.mecanum_pub.publish(recovery_twist)

                        self.pid.clear()
                
                ### 시간 측정 시작 ###
                t3_end = time.time()
                ### 시간 측정 종료 ###
                
                ### 시간 측정 시작 ###
                # 3. 결과 이미지 처리 시간 측정
                t4_start = time.time()
                ### 시간 측정 종료 ###
             
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

                ### 시간 측정 시작 ###
                t4_end = time.time()
                ### 시간 측정 종료 ###

            else:
                # [수정] 주행이 끝나면 (self.start가 False이면) 모든 불을 끕니다.
                self.set_led_color(1, 0, 0, 0)
                self.set_led_color(2, 0, 0, 0)
                time.sleep(0.01)

            
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            ### 시간 측정 시작 ###
            # 최종 로깅
            loop_end_time = time.time()
            total_loop_time = (loop_end_time - time_start) * 1000     

            if self.start:
                preprocessing_time = (t2 - t1) * 1000
                lane_logic_time = (t3_end - t3_start) * 1000
                drawing_time = (t4_end - t4_start) * 1000
            
            self.get_logger().info(
                f'[PERF] Total: {total_loop_time:.2f}ms | '
                f'QueueWait: {queue_wait_time:.2f}ms | '
                f'Preproc: {preprocessing_time:.2f}ms | '
                f'LaneLogic: {lane_logic_time:.2f}ms | '
                f'Drawing: {drawing_time:.2f}ms'
            )
            ### 시간 측정 종료 ###

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        # [수정] 메인 루프 종료 시 LED 끄기
        self.set_led_color(1, 0, 0, 0)
        self.set_led_color(2, 0, 0, 0)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


    # Obtain the target detection result
    def get_object_callback(self, msg):
        ### 시간 측정 시작 ###
        callback_start_time = time.time()
        ### 시간 측정 종료 ###

        self.objects_info = msg.objects

        # 이번 프레임의 타겟 횡단보도를 초기화
        self.target_crosswalk = None 
        self.traffic_signs_status = None

        # 2. 객체 리스트를 한 번만 순회하며 필요한 정보를 모두 추출
        crosswalks_detected = []
        for obj in self.objects_info:
            if obj.class_name == 'crosswalk':
                crosswalks_detected.append(obj)
            elif obj.class_name in ['red', 'green']:
                # Red 신호를 더 높은 우선순위로 처리
                if self.traffic_signs_status is None or self.traffic_signs_status.class_name == 'green':
                    self.traffic_signs_status = obj

        # 3. 추출된 정보를 바탕으로 최종 판단
        # "교차로를 통과 중이 아닐 때" 그리고 "횡단보도가 2개 이상 보일 때"만 타겟 설정
        if not self.is_passing_intersection and len(crosswalks_detected) >= 1:
            # # 가장 큰 횡단보도를 타겟으로 설정
            # self.target_crosswalk = max(crosswalks_detected, key=lambda cw: abs(cw.box[2] - cw.box[0]) * abs(cw.box[3] - cw.box[1]))
             # --- 핵심 수정: 면적이 가장 큰'이 아니라 'y좌표가 가장 큰(가장 가까운)' 횡단보도를 타겟으로 설정 ---
            self.target_crosswalk = max(crosswalks_detected, key=lambda cw: (cw.box[1] + cw.box[3]) / 2)


        # if self.objects_info == []:  # If it is not recognized, reset the variable
        #     pass
        #    # self.crosswalk_distance = 0  # 거리 노필요
        # else:
        #     # min_distance = 0 # 거리 노필요
        #     for i in self.objects_info:
        #         class_name = i.class_name
        #         center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
        #         if class_name == 'crosswalk':
        #             # 횡단보도 부분
        #             crosswalks = [obj for obj in self.objects_info] # if obj.class_name == 'crosswalk']
        #             if len(crosswalks) >= 2 and not self.passed_first_crosswalk:
                
        #                 # 가장 큰 바운딩 박스를 가진 횡단보도를 찾는다
        #                 largest_crosswalk = None
        #                 max_area = 0
        #                 for cw in crosswalks:
        #                     box = cw.box
        #                     area = abs(box[2] - box[0]) * abs(box[3] - box[1])
        #                     if area > max_area:
        #                         max_area = area
        #                         largest_crosswalk = cw
                        
        #                 # 찾은 횡단보도를 이번 프레임의 "주요 타겟"으로 설정
        #                 self.target_crosswalk = largest_crosswalk

        #             # 횡단보도 부분 추가했으므로 이 로직 주석처리
        #             # if center[1] > min_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
        #             #     min_distance = center[1]
                
        #         elif class_name == 'right':  # obtain the right turning sign
        #             self.count_right += 1
        #             self.count_right_miss = 0
        #             if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
        #                 self.turn_right = True
        #                 self.count_right = 0
        #         elif class_name == 'park':  # obtain the center coordinate of the parking sign
        #             self.park_x = center[0]
        #         elif class_name == 'red' or class_name == 'green':  # obtain the status of the traffic light
        #             self.traffic_signs_status = i
               

        #     self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
        #     # self.crosswalk_distance = min_distance # 거리 노필요
        

        # 위 코드 활용하기 위해 추가했던 코드 주석처리
        # 횡단보도 관련 콜백 변수들 추가
            # 1. 횡단보도 객체만 따로 리스트에 저장
            # crosswalks = [obj for obj in self.objects_info if obj.class_name == 'crosswalk']
            
            
            
            # # . 조건 확인: 횡단보도가 2개 이상이고, 아직 첫 번째 횡단보도를 통과하기 전인가?
            # if len(crosswalks) >= 2 and not self.passed_first_crosswalk:
                
            #     # 가장 큰 바운딩 박스를 가진 횡단보도를 찾는다
            #     largest_crosswalk = None
            #     max_area = 0
            #     for cw in crosswalks:
            #         box = cw.box
            #         area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            #         if area > max_area:
            #             max_area = area
            #             largest_crosswalk = cw
                
            #     # 찾은 횡단보도를 이번 프레임의 "주요 타겟"으로 설정
                # self.target_crosswalk = largest_crosswalk

            # (신호등 등 다른 객체 정보 처리 로직은 필요시 여기에 추가)
            # 예시: self.traffic_signs_status = next((obj for obj in self.objects_info if obj.class_name in ['red', 'green']), None)

        callback_end_time = time.time()
        processing_time = (callback_end_time - callback_start_time) * 1000  # 밀리초(ms) 단위
        self.get_logger().info(f'[PERF] Object callback processing time: {processing_time:.2f} ms')
        ### 시간 측정 종료 ###

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.get_logger().info('노드를 종료합니다. LED를 끄고 로봇을 정지합니다.')
        node.set_led_color(1, 0, 0, 0)
        node.set_led_color(2, 0, 0, 0)
        node.mecanum_pub.publish(Twist())
        node.destroy_node()
 
if __name__ == "__main__":
    main()