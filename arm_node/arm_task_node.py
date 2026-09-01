"""
============================================================================
 arm_node.arm_task_node — 任務狀態機
============================================================================
 職責：視覺觸發 → 接近 → 夾取 → 回家，把 controller / scene_builder /
 math_utils 串起來成為一個 ROS2 Node。對應 vision 端 vision_node.py 的角色。

 點雲轉發 / 過濾已搬到視覺端 (vision_node)，本模組不再直接碰點雲。
 clear_octomap 統一在「要去精定位」的當下呼叫一次（見 _move_to_fine）。
============================================================================
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor           # 一邊動、一邊聽 YOLO
from rclpy.callback_groups import ReentrantCallbackGroup    # 允許回呼並行，避免卡死

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Empty  # ★ 用來觸發 octomap 清空重建（精定位前抓一次點雲快照）
from tf2_ros import Buffer, TransformListener, TransformException

from . import config
from .controller import TM5MController
from .math_utils import MathUtils


class TM5MTaskNode(Node):
    def __init__(self):
        super().__init__('tm5m_task_node')
        self.cb_group = ReentrantCallbackGroup()
        self.arm = TM5MController(self, self.cb_group)

        # ★ TF 監聽：用來查法蘭面「現在」實際朝向，供 h 的參考方向使用
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_sub = self.create_subscription(
            PoseStamped, '/target_pose', self.target_callback, 10, callback_group=self.cb_group)
        self.vision_status_sub = self.create_subscription(
            String, '/vision_status', self.vision_status_callback, 10, callback_group=self.cb_group)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10, callback_group=self.cb_group)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self._joint_pos_map   = {name: 0.0 for name in config.ARM_JOINT_NAMES}
        self.is_moving        = True
        self.current_step     = 'INIT'
        self.pause_timer      = None
        self.grasp_target     = None
        self.approach_target  = None
        self.scanning          = False

        self.get_logger().info('大腦節點啟動！等待 MoveIt Server 連線...')
        self.startup_timer = self.create_timer(0.5, self._check_startup)
        self._status_timer = self.create_timer(0.5, self._republish_status)

    def _republish_status(self):
        self.status_pub.publish(String(data='DONE' if self.scanning else 'BUSY'))

    def _check_startup(self):
        if not self.arm.is_ready():
            self.get_logger().info('等待 Server...', throttle_duration_sec=2.0)
            return
        self.startup_timer.cancel()
        self.get_logger().info('所有 Server 已連線！正在建立 Planning Scene...')
        self.arm.control_gripper(config.GRIPPER_RELEASE)
        self.arm.load_environment()
        self.get_logger().info('正在移動至初始姿態...')
        self._move_to_initial()

    def _joint_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_pos_map[name] = pos

    @property
    def current_joints(self):
        return [self._joint_pos_map[name] for name in config.ARM_JOINT_NAMES]

    def target_callback(self, msg: PoseStamped):
        if not self.scanning:
            return
        self.scanning = False
        self._process_target(msg)

    def vision_status_callback(self, msg: String):
        if msg.data != 'NO_TARGET':
            return
        if not self.scanning:
            return
        self.scanning = False
        self.get_logger().info('vision 回報這輪沒有目標，準備回初始位置。')
        self._finish_this_round()

    def _get_current_flange_rotation(self):
        """查 world -> flange 的 TF(URDF 運動鏈用目前 joint_states 做 FK 算出來的)，
        回傳 3x3 旋轉矩陣(欄=局部XYZ軸，世界座標表示)，查不到就回 None，
        呼叫端會自動退回舊版『水平朝向目標』當備用參考，不會整個崩潰。"""
        try:
            t = self.tf_buffer.lookup_transform('world', config.EEF_LINK, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f'查法蘭面目前朝向失敗({e})，退回水平參考。')
            return None
        q = t.transform.rotation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(quat)
        if not np.all(np.isfinite(quat)) or norm < 1e-6:
            self.get_logger().warn('查到的法蘭面朝向四元數非法，退回水平參考。')
            return None
        return R.from_quat(quat / norm).as_matrix()

    def _process_target(self, msg: PoseStamped):
        """實際把單一目標算成 grasp/approach pose 並觸發手臂動作。"""
        pos, q = msg.pose.position, msg.pose.orientation

        # ★ 借用 orientation 的 x, y, z 欄位傳遞 3D 方向向量
        stem_vec = np.array([q.x, q.y, q.z], dtype=float)
        norm = np.linalg.norm(stem_vec)

        if not np.all(np.isfinite(stem_vec)) or norm < 1e-6:
            self.get_logger().warn(f'果梗方向向量非法 (norm={norm:.4f})，放棄這顆，重新掃描。')
            self._enter_scanning()
            return

        stem_vec /= norm
        base_yaw = math.atan2(pos.y, pos.x)

        # ★ 查法蘭面現在實際朝向，取代原本純用 base_yaw 算出來的水平參考 h。
        #   查不到時 calculate_grasp_and_approach 內部會自動退回 base_yaw 版本，
        #   不會崩潰。這裡只換 h 的來源，z_axis 仍是自由投影(沒有鉸鏈限制)。
        R_current = self._get_current_flange_rotation()

        self.get_logger().info(
            f'\n果梗抓取點 X:{pos.x:.3f}, Y:{pos.y:.3f}, Z:{pos.z:.3f} | '
            f'果梗向量:[{stem_vec[0]:.3f}, {stem_vec[1]:.3f}, {stem_vec[2]:.3f}], '
            f'R_current={"查到" if R_current is not None else "查不到,用備用水平朝向"}')

        # 呼叫向量幾何工具算正交姿態
        self.grasp_target, self.approach_target = MathUtils.calculate_grasp_and_approach(
            pos.x, pos.y, pos.z, stem_vec=stem_vec, base_yaw=base_yaw, R_current=R_current)

        self.current_step = 'APPROACH'
        self.is_moving = True
        self.arm.control_gripper(config.GRIPPER_PREOPEN)
        self.get_logger().info('[步驟 1] 往預備點 A ...')
        self.arm.go_to_pose(self.approach_target, done_cb=self.on_action_completed)

    def on_action_completed(self, success):
        if not success:
            self.get_logger().warn('動作執行失敗！')
            if self.current_step == 'INIT':
                self.get_logger().error('回初始姿態也失敗，停止重試，請人工檢查！')
                self._reset_to_idle()
            else:
                self.get_logger().warn('嘗試退回初始姿態...')
                self.arm.control_gripper(config.GRIPPER_RELEASE)
                self.current_step = 'INIT'
                self._move_to_initial()
            return

        d = [math.degrees(j) for j in self.current_joints]
        self.get_logger().info(
            f'當前角度: J1={d[0]:.1f}° J2={d[1]:.1f}° J3={d[2]:.1f}° '
            f'J4={d[3]:.1f}° J5={d[4]:.1f}° J6={d[5]:.1f}°')

        step = self.current_step

        if step == 'INIT':
            self.get_logger().info('已回到初始位置，前往精定位姿態...')
            self._move_to_fine()

        elif step == 'TO_FINE':
            self.get_logger().info(f'已抵達精定位！停頓 {config.PAUSE_BEFORE_SCAN}s 等畫面穩定後開始掃描...')
            self._start_timer(config.PAUSE_BEFORE_SCAN, self._enter_scanning)

        elif step == 'APPROACH':
            self.get_logger().info('抵達點 A，準備下探...')
            self._start_timer(config.PAUSE_AT_APPROACH, self._step_descend)

        elif step == 'DESCEND':
            self.get_logger().info('[步驟 3] 抵達 Goal！閉合夾爪')
            self.arm.control_gripper(config.GRIPPER_GRASP)
            self._start_timer(config.PAUSE_AFTER_GRASP, self._step_lift)

        elif step == 'LIFT':
            if config.GO_TO_BASKET:
                self.get_logger().info('[步驟 5] 退回點 A 完成！準備前往籃子')
                self._step_to_basket()
            else:
                self.get_logger().info('[步驟 5] 這顆處理完成！清點雲、回精定位重新掃描')
                self._move_to_fine()

        elif step == 'BASKET':
            self.get_logger().info('[步驟 7] 抵達籃子上方！放開夾爪')
            self.arm.control_gripper(config.GRIPPER_RELEASE)
            self._start_timer(config.PAUSE_AFTER_RELEASE, self._move_to_fine)

        elif step == 'RETURN':
            self.get_logger().info('已回到初始位置，等手臂穩定後再前往精定位...')
            self._start_timer(config.PAUSE_BEFORE_IDLE, self._final_stabilized_reset)

    def _move_to_initial(self):
        self.is_moving = True
        target = [math.radians(deg) for deg in config.POSE_HOME_DEG]
        self.arm.go_to_joints(target, done_cb=self.on_action_completed)

    def _move_to_fine(self):
        # ★ 只要「要去精定位」，不管從哪個分支呼叫過來，一律先清空 OctoMap，
        #   不用等到達、不依賴 current_step 判斷。
        if self.arm.clear_octomap_client.service_is_ready():
            self.arm.clear_octomap_client.call_async(Empty.Request())
        else:
            self.get_logger().warn('clear_octomap 服務尚未就緒，跳過清空。')
        self.current_step = 'TO_FINE'
        self.is_moving = True
        target = [math.radians(deg) for deg in config.POSE_FINE_DEG]
        self.arm.go_to_joints(target, done_cb=self.on_action_completed)

    def _enter_scanning(self):
        self.scanning = True
        self.is_moving = False
        self.status_pub.publish(String(data='DONE'))

    def _finish_this_round(self):
        self._step_return_home()

    def _step_descend(self):
        self.current_step = 'DESCEND'
        self.get_logger().info('[步驟 2] 暫停結束！筆直前戳到 Goal')
        self.arm.execute_cartesian_path(self.grasp_target, done_cb=self.on_action_completed)

    def _step_lift(self):
        self.current_step = 'LIFT'
        self.get_logger().info('[步驟 4] 夾取完成！原路退回點 A')
        self.arm.execute_cartesian_path(self.approach_target, done_cb=self.on_action_completed)

    def _step_to_basket(self):
        self.current_step = 'BASKET'
        self.get_logger().info('[步驟 6] 前往籃子上方...')
        target = [math.radians(deg) for deg in config.POSE_BASKET_DEG]
        self.arm.go_to_joints(target, done_cb=self.on_action_completed)

    def _step_return_home(self):
        self.current_step = 'RETURN'
        self.get_logger().info('[步驟 8] 回到初始位置')
        self._move_to_initial()

    def _final_stabilized_reset(self):
        self.get_logger().info('手臂穩定完畢！前往精定位，準備下一輪掃描。')
        self.arm.control_gripper(config.GRIPPER_RELEASE)
        self._move_to_fine()

    def _reset_to_idle(self):
        self.arm.control_gripper(config.GRIPPER_RELEASE)
        self.is_moving = False
        self.current_step = 'IDLE'
        self.status_pub.publish(String(data='DONE'))

    def _start_timer(self, duration, callback):
        if self.pause_timer:
            self.pause_timer.cancel()

        def timer_wrapper():
            self.pause_timer.cancel()
            callback()

        self.pause_timer = self.create_timer(duration, timer_wrapper)


def main(args=None):
    rclpy.init(args=args)
    node = TM5MTaskNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
