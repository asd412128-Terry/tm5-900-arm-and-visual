"""進入點：啟動 VisionNode。"""

import cv2
import rclpy
from .vision_node import VisionNode

"""建立並啟動 VisionNode，直到收到 Ctrl+C 或節點關閉為止。"""
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
