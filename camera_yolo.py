import rclpy
# 引入「幾何訊息」格式 「點座標」、「座標轉換」、「姿勢（位置+角度）」
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped
# 引入「感測器訊息」格式 「影像」、「相機資訊」
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from rclpy.node import Node
# ROS的影像格式跟OpenCV影像格式不同，這是一個「翻譯橋樑」。
from cv_bridge import CvBridge

import cv2
import numpy as np
import math
# 多執行緒 一邊處理影像，一邊等你在鍵盤輸入指令。
import threading
import time
# 用來控制電腦系統和終端機（命令提示字元）的輸入行為。
import sys
import termios
# YOLO 函式庫
from ultralytics import YOLO
# 載入處理 TF (Transform) 座標轉換的相關工具
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
# 專門幫幾何點位進行座標轉換的工具
import tf2_geometry_msgs

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()                                      
        
        self.get_logger().info('正在載入 YOLOv11 番茄辨識模型...')
        self.model = YOLO('/home/teerry/yolo_tomato_complex/runs/detect/tomato_m_v3/weights/best.pt')  
        # 準備一個「靜態廣播器」，用來告訴系統相機和手臂之間的固定位置關係。
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        # 呼叫自己寫的函數，發布座標架構
        self.make_tf_bridge()
        self.make_camera_tf()                                         
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self) 
        # 接收彩色影像，收到後交給 color_callback 處理
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        # 接收深度影像，收到後交給 depth_callback 處理
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        # 接收相機參數，收到後交給 info_callback 處理
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        # 接收手臂狀態，收到後交給 status_callback
        self.status_sub = self.create_subscription(String, '/robot_status', self.status_callback, 10)
        # 把算好的目標位置發送到 /target_pose 頻道
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        self.task_completed = False 
        self.camera_info = None        
        self.latest_depth_img = None  
        self.is_asking_user = False   
        self.latest_targets = []      
        
        self.get_logger().info('YOLOv11 視覺大腦啟動！畫面掃描中...')

    def status_callback(self, msg):
        # 如果手臂說 (DONE) ，就把狀態改成 True。
        if msg.data == 'DONE':
            self.task_completed = True

    # 把人類角度轉成機器人角度的公式。
    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw
    
    # 定義「世界 (world)」跟「手臂基座 (base)」的關係
    def make_tf_bridge(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = 0.0, 0.0, 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    # 定義「手臂末端 (link_6)」跟「相機 (camera_optical_frame)」的相對位置
    def make_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_optical_frame'
        
        # 這裡的 0.075 和 0.0521 如果跟 Isaac Sim 裡的真實掛載位置有落差，X 或 Y 就會產生固定誤差
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.075
        t.transform.translation.z = 0.0521
        
        roll, pitch, yaw = 0.0, 0.0, 0.0
        qx, qy, qz, qw = self.euler_to_quaternion(math.radians(roll), math.radians(pitch), math.radians(yaw))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
        self.tf_static_broadcaster.sendTransform(t)

    # 收到相機參數，就把它存到變數裡備用
    def info_callback(self, msg):
        self.camera_info = msg   
    # 把 ROS 傳來的深度訊息，翻譯成 OpenCV 的影像矩陣存起來
    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"深度影像轉換失敗: {e}")

    def color_callback(self, msg):
        # 如果還沒收到相機參數或深度影像，就先跳出不處理
        if self.camera_info is None or self.latest_depth_img is None: return 
        # 去查詢此時此刻，「相機」在「世界地圖」中的絕對座標在哪裡
        try:
            trans = self.tf_buffer.lookup_transform('world', 'camera_optical_frame', rclpy.time.Time()) 
        except TransformException: return 
        # 把接收到的 ROS 彩色影像翻譯成 OpenCV 可視影像
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
        except Exception: return
        
        # 進行 YOLO 預測 (如果還是抓不到，可以把 conf 調降到 0.1 或 0.2)
        results = self.model.predict(cv_image, imgsz = 1280,conf = 0.5, verbose = False)

        detected_objects = [] 
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        ux_img, uy_img = self.camera_info.k[2], self.camera_info.k[5]

        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy() # 取得框框的左上角和右下角座標
                cx_pixel = int((b[0] + b[2]) / 2)
                cy_pixel = int((b[1] + b[3]) / 2)

                if 0 <= cx_pixel < self.latest_depth_img.shape[1] and 0 <= cy_pixel < self.latest_depth_img.shape[0]:
                    
                    # 定義一個 11x11 的小視窗，比大小確保視窗超出影像邊界時不會出錯
                    window = 5
                    y_min = max(0, cy_pixel - window)
                    y_max = min(self.latest_depth_img.shape[0], cy_pixel + window + 1)# shape[0] 是影像高度
                    x_min = max(0, cx_pixel - window)
                    x_max = min(self.latest_depth_img.shape[1], cx_pixel + window + 1)# shape[1] 是影像寬度
                    # 切出中心區域的深度值, 只抓取大於 0 的有效值 (排除掉量不到深度的黑洞)
                    depth_roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                    valid_depths = depth_roi[depth_roi > 0]
                    
                    if len(valid_depths) > 0:
                        z_real = float(np.median(valid_depths))
                        if z_real > 10.0: z_real /= 1000.0 # 單位換算為公尺

                        if z_real > 0.01:
                            # yolo抓到的番茄框框大小
                            w_pixel = b[2] - b[0] # 框框的寬度
                            h_pixel = b[3] - b[1] # 框框的高度
                            avg_pixel_size = (w_pixel + h_pixel) / 2.0 # 照片的番茄直徑
                            f_avg = (fx + fy) / 2.0 # 相機的平均焦距
                            tomato_radius = (avg_pixel_size * z_real) / f_avg / 2.0 # 實際的番茄半徑
                            # 公式 : 實際的番茄直徑/照片的番茄直徑 = 相機焦距/相機到番茄的距離

                            # 將表面深度往後推一個半徑，到達「番茄中心點」
                            z_center = z_real + tomato_radius

                            # 在相機座標系 (Camera Frame) 下計算厚度，再交給 TF 轉換
                            # 這樣不管 J1 轉到幾度，厚度都會自動正確分配給 World 的 X 與 Y！
                            local_point = PointStamped() 
                            local_point.header.frame_id = 'camera_optical_frame'       
                            local_point.header.stamp = self.get_clock().now().to_msg() 
                            
                            # 相機座標系的番茄座標  
                            local_point.point.x = -float((cx_pixel - ux_img) * z_center / fx)
                            local_point.point.y = -float((cy_pixel - uy_img) * z_center / fy)
                            local_point.point.z = float(z_center) 
                            
                            # 機械手臂座標系的番茄座標
                            world_point = tf2_geometry_msgs.do_transform_point(local_point, trans) 
                            #safety_margin = 0.015 # 懸空安全距離 1.5 cm
                            world_point.point.z += (tomato_radius + 0.008)
                            #world_point.point.z += tomato_radius
                            detected_objects.append({
                                'bbox': b, 
                                'cx': cx_pixel, 'cy': cy_pixel,
                                'z_real': z_real,'z_center': z_center, # 記錄中心深度                               
                                'world_x': world_point.point.x,
                                'world_y': world_point.point.y,
                                'world_z': world_point.point.z  
                            })
        
        if len(detected_objects) > 0:
            # sorted(要排誰, 根據什麼排)排序函式
            detected_objects = sorted(detected_objects, key=lambda obj: obj['z_real'])
            self.latest_targets = detected_objects 
            # enumerate : 同時給兩個變數, 索引值 (idx)，對應的物件 (obj) 
            for idx, obj in enumerate(detected_objects):
                b = obj['bbox']
                # 畫出綠色框框, 框住番茄
                cv2.rectangle(cv_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                # 畫出藍色實心圓點, 標示番茄的中心點
                cv2.circle(cv_image, (obj['cx'], obj['cy']), 5, (255, 0, 0), -1)
                # 畫面顯示表面深度，讓你知道相機距離
                cv2.putText(cv_image, f"ID: {idx} Tomato {obj['z_real']:.3f}m", (int(b[0]), int(b[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not self.is_asking_user:
                self.is_asking_user = True
                threading.Thread(target=self.ask_user_thread).start()

        cv2.imshow("YOLOv11 Realtime Vision", cv_image)
        cv2.waitKey(1)
        
    def ask_user_thread(self):
        targets = self.latest_targets
        print("\n" + "="*50) 
        print(f"YOLO 視覺大腦偵測到 {len(targets)} 個番茄目標！")
        for idx, t in enumerate(targets):
            print(f"  [ID: {idx}] 座標 -> X: {t['world_x']:.3f}, Y: {t['world_y']:.3f}, Z: {t['world_z']:.3f}m")
        print("="*50)

        while True:
            # 這行程式碼的作用是清除終端機中尚未處理的輸入，確保在等待使用者輸入時不會有舊的輸入干擾。
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            user_input = input("請輸入要夾取的番茄 ID (輸入 r 重新掃描): ").strip().lower()
            
            # 按r, 結束這個函式，回到主程式重新辨識
            if user_input == 'r':
                self.is_asking_user = False
                return
            
            # 檢查輸入是不是數字，且數字有沒有在番茄清單的範圍內
            if user_input.isdigit() and 0 <= int(user_input) < len(targets):
                selected_id = int(user_input)
                target = targets[selected_id]
                
                distance_to_base = math.sqrt(target['world_x']**2 + target['world_y']**2 + target['world_z']**2)
                max_reach = 0.9

                if distance_to_base > max_reach:
                    print("\n" + "!"*50)
                    print(f" 目標距離基座 {distance_to_base:.3f} 公尺 ，已經超過安全工作範圍")
                    #print(f" 已經超過安全工作範圍 ({max_reach} 公尺)，強行夾取可能會導致 IK 運算失敗或手臂卡死。")
                    #print(" 請選擇其他較近的番茄，或輸入 r 重新掃描。")
                    print("!"*50 + "\n")
                    continue  # 跳過下面的發送指令，回到迴圈開頭讓使用者重新輸入

                print("\n" + "*"*40)
                print(f"目標鎖定！計算含厚度的中心深度 Z: {target['z_center']:.3f} 公尺")
                print(f"發送絕對座標: X={target['world_x']:.3f}, Y={target['world_y']:.3f}")
                print("*"*40)

                try:
                    # 讀取目前手臂最後一軸 (link_6) 的姿勢
                    link6_trans = self.tf_buffer.lookup_transform('world', 'link_6', rclpy.time.Time())
                    # 把目前手臂末端的角度，直接套用到目標點上，這樣不管 J1 轉到幾度，厚度都會自動正確分配給 World 的 X 與 Y！
                    current_orientation = link6_trans.transform.rotation
                except Exception as e:
                    print(f"無法取得手臂角度: {e}")
                    self.is_asking_user = False
                    return
                
                # 填寫要發送給手臂的消息包，包含目標位置和姿勢
                target_msg = PoseStamped()
                target_msg.header.stamp = self.get_clock().now().to_msg()
                target_msg.header.frame_id = 'world'
                target_msg.pose.position.x = target['world_x']  
                target_msg.pose.position.y = target['world_y']
                target_msg.pose.position.z = target['world_z'] 
                target_msg.pose.orientation = current_orientation 
                    
                self.target_pub.publish(target_msg)
                
                print("夾取指令已發出！等待手臂完成動作...")
                self.task_completed = False 
                while not self.task_completed:
                    time.sleep(1.0) 
                
                print("\n手臂動作完成！重新啟動 YOLO 掃描...\n")
                self.is_asking_user = False 
                return
            else:
                print("無效的 ID。")

def main(args=None):
    rclpy.init(args=args)          
    node = VisionNode()            
    try: 
        rclpy.spin(node)          
    except KeyboardInterrupt: 
        pass 
    finally:
        node.destroy_node()        
        cv2.destroyAllWindows()    
        # 修正 4：避免關閉程式時出現 rcl_shutdown already called 的紅字報錯
        if rclpy.ok():
            rclpy.shutdown()           

if __name__ == '__main__':          
    main()