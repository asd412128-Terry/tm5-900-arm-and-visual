#!/usr/bin/env python3
"""
checkerboard_pose_publisher.py
================================
訂閱 RealSense 彩色影像與內參，即時偵測棋盤格姿態，
並把結果以 TF 廣播出去，供 easy_handeye2 取用。

*** 請依實際環境修改下面標 TODO 的參數 ***

用法：
  ros2 run <your_package> checkerboard_pose_publisher.py
  或直接：
  python3 checkerboard_pose_publisher.py
    （前提：這個檔案有被你的 ROS2 package 正確安裝、或先在終端機
      source 好環境後直接跑也可以，只是這樣沒有走 ros2 run 的管理機制）
"""

import os
import numpy as np
import cv2
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped


# ----------------------------------------------------------------------
# TODO: 依你們實際的棋盤格規格修改
# ----------------------------------------------------------------------
CHECKERBOARD_ROWS = 7          # 內角點列數（直向，格子數 8 - 1 = 7）
CHECKERBOARD_COLS = 10          # 內角點行數（橫向，格子數 11 - 1 = 10）
SQUARE_SIZE_M = 0.025          # 每格邊長，單位：公尺（務必量準，量錯會系統性偏移）

# TODO: 依你們 launch 檔裡設定的 frame 名稱修改，要跟 easy_handeye2 的
# tracking_base_frame / tracking_marker_frame 一致
CAMERA_OPTICAL_FRAME = 'camera_color_optical_frame'
CHECKERBOARD_FRAME = 'checkerboard_frame'

# TODO: 依你們 RealSense 實際發布的 topic 名稱修改
COLOR_IMAGE_TOPIC = '/camera/camera/color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'

# 手動校正過的內參檔（ost.yaml 格式），取代相機出廠發布的內參
# 由 ROS2 camera_calibration (cameracalibrator) 產生
CALIBRATION_FILE_PATH = os.path.expanduser('~/tm_ws/calibration/d435i_rgb_calib.yaml')


class CheckerboardPosePublisher(Node):
    def __init__(self):
        super().__init__('checkerboard_pose_publisher')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # 啟動時直接讀入手動校正過的內參，取代相機出廠發布的內參
        self.load_calibration_file(CALIBRATION_FILE_PATH)

        # 棋盤格角點的 3D 世界座標（以棋盤格左上角為原點，Z=0 平面）
        objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
        self.objp = objp * SQUARE_SIZE_M

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)
        self.create_subscription(Image, COLOR_IMAGE_TOPIC, self.image_callback, 10)

        self.get_logger().info('等待相機內參與影像...')

    def load_calibration_file(self, path: str):
        """從 ROS camera_calibration 產生的 ost.yaml 讀入手動校正的 K / D。"""
        try:
            with open(path, 'r') as f:
                calib = yaml.safe_load(f)

            k_flat = calib['camera_matrix']['data']
            d_flat = calib['distortion_coefficients']['data']

            self.camera_matrix = np.array(k_flat, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(d_flat, dtype=np.float64)

            self.get_logger().info(
                f'已從校正檔載入內參：{path}\n'
                f'K =\n{self.camera_matrix}\nD = {self.dist_coeffs}'
            )
        except Exception as e:
            self.camera_matrix = None
            self.dist_coeffs = None
            self.get_logger().error(
                f'讀取校正檔失敗（{path}）：{e}\n'
                '將暫時退回等待 camera_info topic 提供內參（未校正的出廠值）。'
            )

    def camera_info_callback(self, msg: CameraInfo):
        # 校正檔已經在啟動時載入 K/D，這裡不再用 topic 的出廠值覆蓋。
        # 只有在校正檔讀取失敗時，才退回使用相機發布的出廠內參，
        # 確保節點至少還能動作，但這種情況下精度沒有校正過。
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().warn(
                '校正檔未載入，退回使用相機出廠內參（未校正）：\n'
                f'{self.camera_matrix}'
            )

    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return  # 還沒拿到內參，先跳過

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 轉換失敗: {e}')
            return

        self.get_logger().info('收到影像幀，正在偵測...', throttle_duration_sec=1.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        self.get_logger().info(
            f'偵測結果 found={found}  棋盤格設定=({CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS})',
            throttle_duration_sec=1.0
        )

        # debug：每隔一段時間存一張原始灰階圖，方便離線測試不同參數
        if not hasattr(self, '_saved_debug_frame'):
            cv2.imwrite('/tmp/checkerboard_debug.png', gray)
            self._saved_debug_frame = True
            self.get_logger().info('已存一張測試圖到 /tmp/checkerboard_debug.png')

        # debug：不管有沒有偵測到，都先跳窗顯示，確認畫面本身有沒有問題
        cv2.imshow('checkerboard detection', frame)
        cv2.waitKey(1)

        if not found:
            return  # 這幀沒偵測到棋盤格，跳過即可，不用當錯誤處理

        # 角點細化，提升精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        success, rvec, tvec = cv2.solvePnP(
            self.objp, corners_refined, self.camera_matrix, self.dist_coeffs
        )

        if not success:
            self.get_logger().warn('solvePnP 失敗，跳過這一幀')
            return

        # 把 rvec (旋轉向量) 轉成旋轉矩陣，再轉成四元數
        rot_matrix, _ = cv2.Rodrigues(rvec)
        quat = self.rotation_matrix_to_quaternion(rot_matrix)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = CAMERA_OPTICAL_FRAME
        t.child_frame_id = CHECKERBOARD_FRAME
        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

        # debug：畫出偵測結果方便你確認有沒有正確抓到角點
        cv2.drawChessboardCorners(frame, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), corners_refined, found)
        cv2.imshow('checkerboard detection', frame)
        cv2.waitKey(1)

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """3x3 旋轉矩陣轉四元數 (x, y, z, w)"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [x, y, z, w]


def main(args=None):
    rclpy.init(args=args)
    node = CheckerboardPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
