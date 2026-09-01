"""
============================================================================
 純辨識測試腳本
============================================================================
 只吃彩色影像 → 跑 YOLO → 畫框，完全不碰深度/座標轉換/遮擋判斷/抓取點這些邏輯。
 用來獨立驗證「YOLO 模型本身」在目前這顆鏡頭下到底有沒有偵測到東西，排除掉
 detector.py/coordinates.py 那條深度計算路徑的干擾，方便定位問題到底出在
 辨識本身還是後面的座標計算。

 跑法（跟 vision_node 共用同一份 config.py，VISION_MODE 開關一樣有效）：
   python3 -m vision_node.simple_camera_test                    # 實機
   VISION_MODE=isaac python3 -m vision_node.simple_camera_test  # Isaac Sim
============================================================================
"""
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from .config import COLOR_TOPIC, MODEL_PATH, STEM_CLASS_ID, TOMATO_CLASS_ID, YOLO_CONF, YOLO_IMGSZ

"""只訂閱彩色影像、跑 YOLO、畫框顯示，不算深度/世界座標/遮擋/抓取點。"""
class SimpleCameraTestNode(Node):

    def __init__(self):
        super().__init__('simple_camera_test')
        self.bridge = CvBridge()
        self.get_logger().info(f'載入模型: {MODEL_PATH}')
        self.model = YOLO(MODEL_PATH)
        self.sub = self.create_subscription(Image, COLOR_TOPIC, self.callback, 10)
        self.get_logger().info(f'訂閱彩色影像: {COLOR_TOPIC}，等待畫面...')

    """收到一幀彩色影像就跑一次 YOLO 推論，把偵測到的框直接畫出來。"""
    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'影像轉換失敗: {e}')
            return

        results = self.model.predict(cv_image, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        r = results[0]

        n_stem = n_tomato = 0
        if r.boxes is not None:
            cls_ids = r.boxes.cls.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for cls_id, box, conf in zip(cls_ids, boxes, confs):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = map(int, box)

                if cls_id == TOMATO_CLASS_ID:
                    color, label = (0, 165, 255), f'tomato {conf:.2f}'
                    n_tomato += 1
                elif cls_id == STEM_CLASS_ID:
                    color, label = (0, 255, 0), f'stem {conf:.2f}'
                    n_stem += 1
                else:
                    continue

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cv_image, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(cv_image, f'stems={n_stem} tomatoes={n_tomato}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Simple Camera Recognition Test', cv_image)
        cv2.waitKey(1)


"""建立並啟動 SimpleCameraTestNode，直到收到 Ctrl+C 或節點關閉為止。"""
def main(args=None):
    rclpy.init(args=args)
    node = SimpleCameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
