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
import logging
# [빵판 LED] gpiozero 라이브러리 import
from gpiozero import LED
import atexit


class SelfDrivingNode(Node):
    def __init__(self, name):
        # rclpy.init() # <<< 이 줄을 반드시 삭제하세요!
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True

        # Python 표준 로거 설정
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            force=True  # 기존 로거 초기화 덮어쓰기
        )

        self.logger = logging.getLogger(name)

        self.pid = pid.PID(0.2, 0.0, 0.1) # D 제어부분 0.1 로 증가
        self.turn_pid = pid.PID(0.2, 0.0, 0.4) # 일반pid 와 turn 전용 pid 구분, turn 의 D부분을 올림
        self.crosswalk_pid = pid.PID(P=0.004, I=0.0001, D=0.006)# 횡단보도 추적용 PID
  
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

        # --- [빵판 LED] GPIO 설정 ---
        try:
            self.breadboard_led = LED(17) # GPIO 17번 핀
            atexit.register(self.breadboard_led.off)
            self.logger.info('빵판 LED용 GPIO 17번 핀을 성공적으로 초기화했습니다.')
        except Exception as e:
            self.logger.error(f'GPIO 초기화 실패: {e}')
            self.breadboard_led = None
        self.is_blinking = False

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
        if self.breadboard_led:
            self.breadboard_led.off()


    
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
        self.logger.info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True
        self.first = True  # for warm-up

   
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
        self.normal_speed = 0.5 # normal driving speed # 횡단보도 인식을 위해 줄이는중..
        self.slow_down_speed = 0.2  # slowing down speed
        self.cornering_speed = 0.2 # 코너링 스피드 ( 미리 진입하기 전부터 작동 )

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []
        self.initial_go_signal_received = False # 최초 출발 신호 수신 여부
        # [추가] "주차 완료" 상태를 초기화합니다.
        self.parking_completed = False

        # 추가 - crosswalk 를 인식하고 정지하기 위한 변수선언 
        self.target_crosswalk = None # 정지 목표가 되는 횡단보도 객체 정보
        self.passed_first_crosswalk = False # 첫 번째 횡단보도를 통과했는지 여부
        self.stop_for_crosswalk_start_time = 0 # 횡단보도 앞 정지 시작 시간
        self.is_passing_intersection = False  # 현재 교차로를 통과/진행 중인지 여부
        self.crosswalk_disappear_counter = 0  # 횡단보도가 안 보이기 시작한 프레임 수

        # 추가
        self.intersection_pass_start_time = 0 # 교차로 통과를 시작한 시간

        # 우회전 관련
        # self.is_performing_hard_turn = False # 하드코딩 우회전 수행 여부
        # self.hard_turn_start_time = 0      # 하드코딩 우회전 시작 시간
        self.right_sign_detected = False   # 'right' 표지판 감지 여부
        self.special_maneuver_stage = 0  # 0:없음, 1:우회전-직진, 2:우회전-회전, 3:횡단보도 추적
        self.maneuver_start_time = 0
        self.crosswalk_steer_kp = 0.005  # 횡단보도 추적 P제어 게인 (튜닝 필요)
        # park
        self.park_sign_detected = False    # 'park' 표지판 감지 여부
        self.is_parking_disabled = False   # 주차 금지 활성화 여부
        self.is_parking_approach_mode = False # 주차 접근 모드에 진입했는지 여부

        # 횡단보도 추적관련
        self.is_far_target_acquired = False # 3단계에서 원거리 목표를 확정했는지 여부
        

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.logger.info('\033[1;32m%s\033[0m' % "self driving enter")
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
        self.logger.info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.logger.info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.logger.info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            with self.lock:
                if self.start != request.data:
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

    # [추가] 주차 완료 LED 깜빡임 효과 함수
    def _parking_complete_effect(self):
        """주차 완료 후 5초간 파란색 LED를 깜빡입니다."""
        self.logger.info("주차 완료! 축하 세리머니를 시작합니다.")
        end_time = time.time() + 5  # 5초 동안 실행
        
        while time.time() < end_time:
            # LED 켜기 (파란색)
            self.set_led_color(1, 0, 0, 255)
            self.set_led_color(2, 0, 0, 255)
            time.sleep(0.25) # 0.25초 대기
            
            # LED 끄기
            self.set_led_color(1, 0, 0, 0)
            self.set_led_color(2, 0, 0, 0)
            time.sleep(0.25) # 0.25초 대기
            
        # 깜빡임이 끝난 후 LED를 확실히 끈다.
        self.set_led_color(1, 0, 0, 0)
        self.set_led_color(2, 0, 0, 0)

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
            self.mecanum_pub.publish(twist)
            time.sleep(0.65/0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())
        if self.breadboard_led and not self.is_blinking:
            self.logger.info("주행 종료: 빵판 LED 5초간 점멸 시작 (백그라운드)")
            # background=True로 설정하여 즉시 다음 코드로 넘어가게 함
            self.breadboard_led.blink(on_time=0.25, off_time=0.25, n=10, background=True)
            self.is_blinking = True
        # 1. 일회성 세리머니 이벤트 실행
        self._parking_complete_effect()

        # 2. 모든 동작이 끝났으므로, '주차 완료' 상태 플래그를 True로 설정
        self.parking_completed = True
        self.logger.info('주차 완료 및 세리머니 종료. 이제부터 LED는 계속 소등됩니다.')
        # [수정] 주차 완료 시, self.start 플래그를 False로 변경하여 주행 종료 상태로 전환
        self.start = False

        # 주차 조건 확인 및 실행
    def _handle_parking(self):

        """주차 조건을 확인하고, 충족 시 주차 동작을 실행합니다."""

        if self.is_parking_disabled:
            return False

        # 주차 표지판이 없거나 이미 주차를 완료했다면 아무것도 하지 않음
        if not self.park_sign_detected or self.parking_completed:
            return False

        # 근처 횡단보도를 기준으로 주차 위치 판단
        crosswalks_detected = [obj for obj in self.objects_info if obj.class_name == 'crosswalk']
        for cw in crosswalks_detected:
            if cw.box[3] > 480 * 0.3:  # 횡단보도가 일정 위치에 오면
                self.logger.info("주차 조건 충족! 주차를 시작합니다.")
                self.park_action()  # 블로킹 방식의 주차 동작 실행
                self.start = False  # 주차 완료 후 자율주행 미션 종료
                return True # 주차를 시작했음을 알림
            
        return False


        # 우회전 실행
    # 우회전 실행
    def _handle_special_maneuver(self):
        """단계별 특별 동작(우회전, 횡단보도 추적)을 수행합니다."""
        twist = Twist()
        elapsed_time = time.time() - self.maneuver_start_time

        # --- 1단계: 우회전 전 직진 ---
        if self.special_maneuver_stage == 1:
            if elapsed_time < 1.3:
                self.logger.info(f"특별 동작 1단계: 직진 ({elapsed_time:.2f}s)")
                twist.linear.x = 0.4
            else: # 1.3초 경과 시 2단계로 전환
                self.logger.info("특별 동작 1단계 완료. 2단계(회전)로 전환합니다.")
                self.special_maneuver_stage = 2
                self.maneuver_start_time = time.time() # 다음 단계를 위해 타이머 초기화

        # --- 2단계: 우회전 ---
        elif self.special_maneuver_stage == 2:
            if elapsed_time < 1.93: # 90도 회전을 위한 시간 (튜닝 필요)
                self.logger.info(f"특별 동작 2단계: 회전 ({elapsed_time:.2f}s)")
                twist.angular.z = -0.8
                
            else: # 1.8초 경과 시 3단계로 전환
                self.logger.info("특별 동작 2단계 완료. 3단계(자세 안정화)로 전환합니다.")
                self.special_maneuver_stage = 3
                self.maneuver_start_time = time.time() # 다음 단계를 위해 타이머 초기화
                
        # --- 3단계: 우회전 후 자세 안정화를 위한 직진 ---
        elif self.special_maneuver_stage == 3:
            if elapsed_time < 2.3:
                self.logger.info(f"특별 동작 3단계: 자세 안정화 직진 ({elapsed_time:.2f}s)")
                twist.linear.x = 0.8 #self.normal_speed
                self.is_parking_disabled = True
            else:
                self.logger.info("특별 동작 3단계 완료. 4단계(주차 접근)로 전환합니다.")
                self.special_maneuver_stage = 4
                self.crosswalk_pid.clear() # PID 제어 시작 전 초기화
                self.is_parking_disabled = False

        # --- 4단계: [수정됨] 주차를 위한 횡단보도 접근 및 정렬 ---
        elif self.special_maneuver_stage == 4:
            crosswalks = [obj for obj in self.objects_info if obj.class_name == 'crosswalk']
            
            # 접근할 횡단보도가 보이지 않으면, 탐색을 위해 느리게 우회전하며 전진
            if not crosswalks:
                self.logger.info("특별 동작 [4/4]: 주차용 횡단보도 탐색 중...")
                twist.linear.x = 0.15
                twist.angular.z = -0.2
                return twist

            # 우회전 후이므로, 주차 지점은 가장 가까운(이미지 아래쪽) 횡단보도임
            nearest_crosswalk = max(crosswalks, key=lambda cw: cw.box[3])
            
            # [최종 조건 확인] 주차 표지판이 보이고, 횡단보도가 충분히 가까워졌다면 주차 실행
            if self.park_sign_detected and nearest_crosswalk.box[3] > 480 * 0.5:
                self.logger.info("!!! 주차 조건 최종 충족! 주차를 시작합니다. !!!")
                self.park_action()
                self.start = False
                self.special_maneuver_stage = 0
                return Twist() # 주차 동작 실행 후 정지
            
            # [접근 로직] 아직 조건이 안되면, 가장 가까운 횡단보도를 향해 PID 제어로 접근
            else:
                log_msg = "Park 감지!" if self.park_sign_detected else "Park 표지판 찾는 중..."
                self.logger.info(f"특별 동작 [4/4]: 가까운 횡단보도로 접근 중... ({log_msg})")
                
                target_x = (nearest_crosswalk.box[0] + nearest_crosswalk.box[2]) / 2
                center_of_image = 320
                self.crosswalk_pid.SetPoint = center_of_image
                self.crosswalk_pid.update(target_x)
                
                twist.angular.z = self.crosswalk_pid.output
                twist.linear.x = self.normal_speed * 0.5
                
        return twist

    def _perform_warmup(self):
        """첫 실행 시 주요 연산 함수를 미리 호출하여 최적화를 준비합니다."""
        self.logger.info("Starting warm-up routine...")

        # 가짜 이미지 생성 (실제 카메라 해상도와 동일하게)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 주요 연산 함수를 미리 호출하여 예열
        for _ in range(5): # 5번 정도 반복
            _ = self.lane_detect.get_binary(dummy_image)
        self.logger.info("Warm-up finished.")
        self.first = False

    def _get_latest_image(self):
        """이미지 큐를 비우고 가장 마지막에 들어온 이미지만 반환합니다."""
        if self.image_queue.empty():
            return None, None

        items_in_queue = []

        # 1. 큐가 빌 때까지 모든 항목을 꺼내서 임시 리스트에 담습니다. (줄 세워 내보내기)
        while not self.image_queue.empty():
            try:
                items_in_queue.append(self.image_queue.get_nowait())
            except queue.Empty:
                break
        
        # 2. 임시 리스트가 비어있으면 처리할 게 없으므로 건너뜁니다.
        if not items_in_queue:
            return None, None
            
        return items_in_queue[-1] # (image, reception_time)

    def _update_leds(self):
                """로봇의 현재 상태에 따라 LED 색상을 업데이트합니다."""
                # 우선순위 0: 주차 완료
                if self.parking_completed:
                    self.set_led_color(1, 0, 0, 0)
                    self.set_led_color(2, 0, 0, 0)
                    return # 다른 LED 상태를 덮어쓰지 않도록 여기서 함수 종료
                # [추가] 우선순위 1: 횡단보도에서 정지
                elif self.stop_for_crosswalk_start_time > 0:
                    self.set_led_color(1, 255, 0, 0)  # 빨간색
                    self.set_led_color(2, 255, 0, 0)  # 빨간색
                # 우선순위 2: 일반 정지 상태 (stop 플래그)
                elif self.stop:
                    self.set_led_color(1, 255, 0, 0)
                    self.set_led_color(2, 255, 0, 0)
                # 우선순위 3: 우회전 중
                elif self.start_turn:
                    self.set_led_color(1, 0, 255, 0) # 1번 LED는 초록색 유지
                    # 깜빡임 로직
                    if time.time() - self.last_blink_time > 0.25:
                        self.last_blink_time = time.time()
                        # 현재 LED 2번 상태를 토글
                        if self.led2_color == (255, 255, 0):
                            # 강제로 초기화 후 전송
                            self.led2_color = (0, 0, 0)
                            msg = RGBStates()
                            state = RGBState()
                            state.index = 2
                            state.red = 0
                            state.green = 0
                            state.blue = 0
                            msg.states.append(state)
                            self.rgb_pub.publish(msg)
                        else:
                            # 강제로 초기화 후 전송
                            self.led2_color = (255, 255, 0)
                            msg = RGBStates()
                            state = RGBState()
                            state.index = 2
                            state.red = 255
                            state.green = 255
                            state.blue = 0
                            msg.states.append(state)
                            self.rgb_pub.publish(msg)
                # 우선순위 4: 일반 주행 상태
                else:
                    self.set_led_color(1, 0, 255, 0)
                    self.set_led_color(2, 0, 255, 0)

    def _wait_for_initial_green_light(self):
        """
        최초 출발 신호를 확인합니다. 녹색 신호등이 감지될 때까지 기다립니다.
        Returns:
            bool: 주행을 시작할 수 있으면 True, 대기해야 하면 False.
        """
        # YOLOv5가 감지한 신호등 객체가 있는지 확인
        if self.traffic_signs_status is not None:
            # 신호등이 초록불이면 출발 허가
            if self.traffic_signs_status.class_name == 'green':
                self.logger.info("출발 신호 (초록불) 감지! 주행을 시작합니다.")
                return True
            # 신호등이 빨간불이면 대기
            elif self.traffic_signs_status.class_name == 'red':
                self.logger.info("정지 신호 (빨간불) 감지. 출발 대기 중...")
                return False
        
        # 신호등이 아직 감지되지 않았으면 계속 대기
        self.logger.info("출발 신호등을 찾는 중... 대기합니다.")
        return False

    def _handle_crosswalks(self):
        """
        횡단보도 및 신호등을 처횡단보도를리하고 정지 여부를 결정합니다.
        (로직 개선 버전)
        """

        # 1. '교차로 통과 중' 상태이면 3초간 횡단보도 인식 무시
        if self.is_passing_intersection:
            # 4초가 지났으면 '교차로 통과' 상태를 해제
            if time.time() - self.intersection_pass_start_time > 4.0:
                self.logger.info("교차로 통과 완료. 다음 횡단보도를 준비합니다.")
                self.is_passing_intersection = False
            else:
                # 3초가 지나지 않았으면, 정지 로직을 건너뛰고 주행 유지
                return False 

        # 2. '교차로 통과 중'이 아닐 때, 정지해야 하는지 판단
        # 현재 프레임에서 감지된 횡단보도만 필터링
        crosswalks_detected = [obj for obj in self.objects_info if obj.class_name == 'crosswalk']

        # 아직 정지하지 않았고, 보이는 횡단보도가 1개 이상일 때
        if self.stop_for_crosswalk_start_time == 0 and len(crosswalks_detected) > 0:
            for cw in crosswalks_detected:
                # 감지된 횡단보도 중 하나라도 화면의 60% 아래로 내려오면 정지
                if cw.box[3] > 480 * 0.6: 
                    self.logger.info("횡단보도 근접, 정지를 시작합니다.")
                    self.mecanum_pub.publish(Twist())  # 정지 명령
                    self.stop_for_crosswalk_start_time = time.time()
                    return True  # 정지했으므로 True 반환

        # 3. 횡단보도 앞에 정지해 있는 동안의 로직 (신호등 또는 시간 대기)
        if self.stop_for_crosswalk_start_time > 0:
            # 우회전 표지판 감지 시 특별 동작을 최우선으로 처리
            if self.right_sign_detected:
                self.logger.info("'right' 표지판 감지! 특별 동작을 시작합니다.")
                # =======================================================
                # ▼ 아래 2줄을 수정합니다.
                self.special_maneuver_stage = 1 # 1단계(우회전-직진) 시작
                self.maneuver_start_time = time.time()
                # =======================================================
                self.stop_for_crosswalk_start_time = 0
                return False



            can_go = False
            
            # 신호등이 감지된 경우
            if self.traffic_signs_status is not None:
                if self.traffic_signs_status.class_name == 'green':
                    self.logger.info("초록불 감지. 출발합니다!")
                    can_go = True
                elif self.traffic_signs_status.class_name == 'red':
                    self.logger.info("빨간불 감지. 대기합니다...")
                    # 정지 상태 유지 (필요 시 정지 명령 재전송)
            # 신호등이 없을 때
            else:
                # 1초 대기 후 출발
                if time.time() - self.stop_for_crosswalk_start_time > 1.2:
                    self.logger.info("신호등 없음. 1초 대기 후 출발합니다.")
                    can_go = True
            
            # 출발 조건이 충족되면
            if can_go:
                self.is_passing_intersection = True  # '교차로 통과 중' 상태로 전환
                self.intersection_pass_start_time = time.time()  # 통과 시작 시간 기록
                self.stop_for_crosswalk_start_time = 0 # 정지 상태 해제
                return False # 이제 주행해야 하므로 False 반환
            else:
                return True # 아직 대기해야 하므로 True 반환

        # 위의 어떤 조건에도 해당하지 않으면 주행 (정지할 필요 없음)
        return False
    
    def _follow_lane(self, binary_image, image):
        twist = Twist()

        # line following processing
        result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
        
        if lane_x >= 0 and not self.stop:
            
            # 1. 급회전 구간 (우회전 시작)
            if lane_x > 180:
                
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

            # 2. 완만한 회전 구간 (코너링 준비)
            elif lane_x > 150: # 코너링 미리 감지하고 준비 ( 속도감소 ) - 2순위
                twist.linear.x = self.cornering_speed
                # 감속 중에도 PID 제어는 유지
                self.pid.SetPoint = 130 
                self.pid.update(lane_x)
                twist.angular.z = -common.set_range(self.pid.output, -0.3, 0.3)
            
            # 3. 직선 구간
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
        else:
            # 차선인식 못한 경우
            self.lane_detect.set_mode("close_up")

            # 느린 명령주면서 차선 찾기
            twist.linear.x = 0.1  # 느린 속도로
            twist.angular.z = 0.4   # 왼쪽으로 살짝 회전
            self.pid.clear()
        
        return result_image, twist
    

    def _draw_overlays(self, image):
        """결과 이미지에 객체 바운딩 박스와 FPS 정보를 그립니다."""
        # 객체 정보 시각화
        if self.objects_info:
            for i in self.objects_info:
                box = i.box
                class_name = i.class_name
                cls_conf = i.score
                cls_id = self.classes.index(class_name)
                color = colors(cls_id, True)
                plot_one_box(
                    box,
                    image,
                    color=color,
                    label="{}:{:.2f}".format(class_name, cls_conf),
                )
        # FPS 정보 시각화
        if self.display:
            self.fps.update()
            image = self.fps.show_fps(image)

        return image
    
    def _log_performance(self, start_time, reception_time, t1_start, t1_end, t2_start, t2_end, t3_start, t3_end):
        """각 단계별 처리 시간을 계산하고 로그로 출력합니다."""
        total_loop_time = (time.time() - start_time) * 1000
        queue_wait_time = (start_time - reception_time) * 1000
        
        if self.start:
            preprocessing_time = (t1_end - t1_start) * 1000
            crosswalk_logic_time = (t2_end - t2_start) * 1000
            lane_logic_time = (t3_end - t3_start) * 1000
            
            # 드로잉 시간은 이제 _draw_overlays 함수에서 별도 측정 가능
            self.logger.info(
                f'[PERF] Total: {total_loop_time:.2f}ms | '
                f'QueueWait: {queue_wait_time:.2f}ms | '
                f'Preproc: {preprocessing_time:.2f}ms | '
                f'Crosswalk: {crosswalk_logic_time:.2f}ms | '
                f'Lane: {lane_logic_time:.2f}ms'
            )
        else:
            self.logger.info(f'[PERF] Total: {total_loop_time:.2f}ms')

    def main(self):
        """메인 루프: 로봇의 상태를 확인하고 적절한 동작을 지시합니다."""
        if self.first:
            self._perform_warmup()
            
        while self.is_running:
            image, reception_time = self._get_latest_image()
            if image is None:
                time.sleep(0.001)
                continue

            result_image = image.copy()
            twist_command = Twist()

            if self.start:
                # --- 빵판 LED 제어 (주행 중) ---
                if self.breadboard_led:
                    if self.is_blinking:
                        self.breadboard_led.off()
                        self.is_blinking = False
                    self.breadboard_led.on()
                # =================== [추가된 로직 시작] ===================
                # 최초 출발 신호 대기 (주행 시작 후 한 번만 실행됨)
                if not self.initial_go_signal_received:
                    if self._wait_for_initial_green_light():
                        # 초록불이 확인되면, 플래그를 True로 바꿔 다시는 이 로직을 실행하지 않음
                        self.initial_go_signal_received = True
                    else:
                        # 아직 출발 신호가 아니면 로봇을 정지시키고, LED를 빨간색으로 변경
                        self.mecanum_pub.publish(Twist()) 
                        self.set_led_color(1, 255, 0, 0) # 대기 중임을 알리는 빨간색 LED
                        self.set_led_color(2, 255, 0, 0)
                        
                        # 화면 출력은 계속 수행
                        result_image = self._draw_overlays(result_image)
                        bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
                        cv2.imshow('Self Driving View', bgr_image)
                        cv2.waitKey(1)
                        
                        # 이후의 주행 로직을 건너뛰고 다음 루프로 넘어감
                        continue
                # =================== [추가된 로직 끝] =====================

                self._update_leds()
                
                # 1. 주차 로직 (최고 우선순위)
                if self._handle_parking():
                    self.mecanum_pub.publish(Twist()) # 주차 완료 후 정지
                    continue # 주차 미션 완료, 이후 동작 중지

                binary_image = self.lane_detect.get_binary(image)
                
                # 2. 우회전 또는 일반 주행 로직 결정
                if self.special_maneuver_stage > 0:
                    twist_command = self._handle_special_maneuver()
                else:
                    # 횡단보도 처리 후 정지해야 하는지 확인
                    should_stop = self._handle_crosswalks()
                    if not should_stop:
                        # 정지할 필요 없으면 차선 주행
                        result_image, twist_command = self._follow_lane(binary_image, result_image)
                
                self.mecanum_pub.publish(twist_command)

            else: # self.start가 False일 때 (주차 완료 또는 시작 전)
                self.set_led_color(1, 0, 0, 0)
                self.set_led_color(2, 0, 0, 0)
                time.sleep(0.01)

            # 시각화 및 결과 전송
            result_image = self._draw_overlays(result_image)
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
            cv2.imshow('Self Driving View', bgr_image)
            cv2.waitKey(1)
                
        # 메인 루프 종료 시 정리
        self.set_led_color(1, 0, 0, 0)
        self.set_led_color(2, 0, 0, 0)
        self.mecanum_pub.publish(Twist())
        # rclpy.shutdown() # <<< 이 줄을 반드시 삭제하

    # Obtain the target detection result
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        self.traffic_signs_status = None
        self.right_sign_detected = False # 매번 콜백이 호출될 때마다 초기화
        self.park_sign_detected = False  # 매번 콜백이 호출될 때마다 초기화

        # 객체 리스트를 순회하며 필요한 정보 업데이트
        for obj in self.objects_info:
            # 'right' 표지판 감지
            if obj.class_name == 'right':
                self.right_sign_detected = True
            
            # 'park' 표지판 감지
            elif obj.class_name == 'park':
                self.park_sign_detected = True
                self.logger.info("주차표시판 인식")

            # 신호등 정보 찾기 (Red 신호 우선)
            elif obj.class_name in ['red', 'green']:
                if self.traffic_signs_status is None or self.traffic_signs_status.class_name == 'green':
                    self.traffic_signs_status = obj

def main():
    rclpy.init() # <<< 여기에 한 번만 init()을 호출해야 합니다.
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.logger.info('노드를 종료합니다. LED를 끄고 로봇을 정지합니다.')
        if node.breadboard_led:
            node.breadboard_led.off()
        node.set_led_color(1, 0, 0, 0)
        node.set_led_color(2, 0, 0, 0)
        node.mecanum_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown() # <<< 여기에 한 번만 shutdown()을 호출해야 합니다.

 
if __name__ == "__main__":
    main()