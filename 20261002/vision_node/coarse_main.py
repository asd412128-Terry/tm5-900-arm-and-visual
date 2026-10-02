"""進入點：啟動 CoarseNode（車載相機粗定位）。python3 -m vision_node.coarse_main"""

import cv2
import rclpy
from .coarse_node import CoarseNode

"""建立並啟動 CoarseNode，直到收到 Ctrl+C 或節點關閉為止。"""
def main(args=None):
    rclpy.init(args=args)
    node = CoarseNode()
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
