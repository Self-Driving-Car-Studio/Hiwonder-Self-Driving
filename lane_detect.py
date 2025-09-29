import os
import cv2
import math
import queue
import threading
import numpy as np
import sdk.common as common
from cv_bridge import CvBridge

bridge = CvBridge()

lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")

class LaneDetector(object):
    def __init__(self, color):
        self.target_color = color

        # 1. 3가지 ROI 탐색 전략 정의
        # 각 전략은 (y_start, y_end, x_start, x_end, weight) 튜플의 집합입니다.
        if os.environ.get('DEPTH_CAMERA_TYPE') == 'ascamera':
            # ascamera (보통 640x360 해상도) 기준
            # 전략 1: 하단 집중 (안정적인 직진)
            self.rois_bottom = ((338, 360, 0, 320, 0.7), (292, 315, 0, 320, 0.2), (248, 270, 0, 320, 0.1))
            # 전략 2: 중단 집중 (일반적인 코너)
            self.rois_middle = ((260, 290, 0, 320, 0.7), (220, 250, 0, 320, 0.2), (180, 210, 0, 320, 0.1))
            # 전략 3: 상단 집중 (먼 거리의 코너 예측)
            self.rois_top = ((180, 210, 0, 320, 0.7), (140, 170, 0, 320, 0.2), (100, 130, 0, 320, 0.1))
        else:
            # 일반 USB 카메라 (보통 640x480 해상도) 기준
            # 전략 1: 하단 집중 (안정적인 직진)
            self.rois_bottom = ((450, 480, 0, 320, 0.7), (390, 420, 0, 320, 0.2), (330, 360, 0, 320, 0.1))
            # 전략 2: 중단 집중 (일반적인 코너)
            self.rois_middle = ((340, 380, 0, 320, 0.7), (290, 330, 0, 320, 0.2), (240, 280, 0, 320, 0.1))
            # 전략 3: 상단 집중 (먼 거리의 코너 예측)
            self.rois_top = ((240, 280, 0, 320, 0.7), (180, 220, 0, 320, 0.2), (120, 160, 0, 320, 0.1))

        self.weight_sum = 1.0

    def set_roi(self, roi):
        # 이 함수는 동적 탐색 로직에서는 직접 사용되지 않지만, 호환성을 위해 남겨둡니다.
        self.rois_bottom = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=30): # 임계값을 30으로 낮춰 작은 차선 조각도 감지
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None
    
    # 헬퍼 함수: 주어진 ROI 세트로 차선을 탐색
    def _find_lane_in_rois(self, image, rois, result_image):
        centroid_sum = 0
        max_center_x = -1
        center_x_list = []

        for roi in rois:
            blob = image[roi[0]:roi[1], roi[2]:roi[3]]
            contours = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            max_contour_area = self.get_area_max_contour(contours)

            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                
                box[:, 0] += roi[2]
                box[:, 1] += roi[0]
                
                cv2.drawContours(result_image, [box], -1, (255, 255, 0), 2)
                
                line_center_x = (box[0][0] + box[2][0]) / 2
                center_x_list.append(line_center_x)
            else:
                center_x_list.append(-1)
        
        valid_detections = False
        for i, center_x in enumerate(center_x_list):
            if center_x != -1:
                valid_detections = True
                if center_x > max_center_x:
                    max_center_x = center_x
                centroid_sum += center_x * rois[i][-1]
        
        if not valid_detections:
            return None, -1 # 차선을 전혀 찾지 못함
        
        center_pos = centroid_sum / self.weight_sum
        return center_pos, max_center_x

    # 기존 함수들은 그대로 유지 (add_horizontal_line, add_vertical_line_far, add_vertical_line_near, get_binary)
    def add_horizontal_line(self, image):
        h, w = image.shape[:2]
        roi = image[0:h, int(w/2):w]
        flip_binary = cv2.flip(roi, 0)
        max_y = cv2.minMaxLoc(flip_binary)[-1][1]
        return h - max_y

    def add_vertical_line_far(self, image):
        h, w = image.shape[:2]
        roi = image[0:h, int(w/8):int(w/2)]
        flip_binary = cv2.flip(roi, -1)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]
        y_center1 = y_0 + 55
        roi1 = flip_binary[y_center1:, :]
        (x_1, y_1) = cv2.minMaxLoc(roi1)[-1]
        down_p = (int(w/2) - x_1, h - (y_1 + y_center1))
        y_center2 = y_0 + 65
        roi2 = flip_binary[y_center2:, :]
        (x_2, y_2) = cv2.minMaxLoc(roi2)[-1]
        up_p = (int(w/2) - x_2, h - (y_2 + y_center2))
        up_point, down_point = (0, 0), (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            k = (up_p[1] - down_p[1]) / (up_p[0] - down_p[0])
            up_point = (int(-down_p[1] / k + down_p[0]), 0)
            down_point = (int((h - down_p[1]) / k + down_p[0]), h)
        return up_point, down_point

    def add_vertical_line_near(self, image):
        h, w = image.shape[:2]
        roi = image[int(h/2):h, 0:int(w/2)]
        flip_binary = cv2.flip(roi, -1)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]
        down_p = (int(w/2) - x_0, h - y_0)
        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        y_center = int((h/2 - y_1 + y_0) / 2)
        roi_up = flip_binary[y_center:, :]
        (x, y) = cv2.minMaxLoc(roi_up)[-1]
        up_p = (int(w/2) - x, h - (y + y_center))
        up_point, down_point = (0, 0), (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            k = (up_p[1] - down_p[1]) / (up_p[0] - down_p[0])
            up_point = (int(-down_p[1] / k + down_p[0]), 0)
            down_point = down_p
        return up_point, down_point, y_center

    def get_binary(self, image):
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
        mask = cv2.inRange(img_blur, tuple(lab_data['lab']['Stereo'][self.target_color]['min']), tuple(lab_data['lab']['Stereo'][self.target_color]['max']))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return dilated

    # __call__ 함수를 동적 탐색 로직으로 교체
    def __call__(self, image, result_image):
        h, w = image.shape[:2]
        
        strategies = {
            "BOTTOM": self.rois_bottom,
            "MIDDLE": self.rois_middle,
            "TOP": self.rois_top,
        }

        final_center_pos = None
        final_max_center_x = -1

        for mode, rois in strategies.items():
            center_pos, max_center_x = self._find_lane_in_rois(image, rois, result_image)
            
            # RQT에 현재 탐색 모드 표시
            cv2.putText(result_image, f"Search Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if center_pos is not None:
                final_center_pos = center_pos
                final_max_center_x = max_center_x
                break # 차선을 찾았으면 탐색 중단
        
        if final_center_pos is None:
            # 모든 전략으로도 차선을 못 찾음
            return result_image, None, -1

        angle = math.degrees(-math.atan((final_center_pos - (w / 2.0)) / (h / 2.0)))
        
        return result_image, angle, final_max_center_x

image_queue = queue.Queue(2)
def image_callback(ros_image):
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    bgr_image = np.array(cv_image, dtype=np.uint8)
    if image_queue.full():
        image_queue.get()
    image_queue.put(bgr_image)

def main():
    running = True

    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running:
                break
            else:
                continue
        binary_image = lane_detect.get_binary(image)
        cv2.imshow('binary', binary_image)
        img = image.copy()
        y = lane_detect.add_horizontal_line(binary_image)
        roi = [(0, y), (640, y), (640, 0), (0, 0)]
        cv2.fillPoly(binary_image, [np.array(roi)], [0, 0, 0]) 
        min_x = cv2.minMaxLoc(binary_image)[-1][0]
        cv2.line(img, (min_x, y), (640, y), (255, 255, 255), 50) 
        result_image, angle, x = lane_detect(binary_image, image.copy()) 
        '''
        up, down = lane_detect.add_vertical_line_far(binary_image)
        #up, down, center = lane_detect.add_vertical_line_near(binary_image)
        cv2.line(img, up, down, (255, 255, 255), 10)
        '''
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    import rclpy
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node('lane_detect')
    lane_detect = LaneDetector('yellow')
    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)
    threading.Thread(target=main, daemon=True).start()
    rclpy.spin(node)




