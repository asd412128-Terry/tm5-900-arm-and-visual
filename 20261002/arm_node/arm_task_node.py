"""
============================================================================
 arm_node.arm_task_node — 任務狀態機
============================================================================
 職責：視覺觸發 → 接近 → 夾取 → 回家，把 controller / scene_builder /
 math_utils 串起來成為一個 ROS2 Node。對應 vision 端 fine_node.py 的角色。

 ★ 2026-09-29：手臂統一指揮哪個視覺節點工作（/robot_status，見 _current_status）：
   確定回到 Home → COARSE（開粗定位）→ 收到座標鎖定 → BUSY → 前往精定位觀測點
   → 確定到達 → DONE（開精定位）→ 夾取 → 回同一個觀測點重掃
   → 精定位看不到 / 任何一步失敗 → 回 Home → COARSE（重新粗定位）

 點雲轉發 / 過濾 / 清空 OctoMap 都在視覺端 (vision_node)，本模組不碰點雲。
 ★ 2026-09-30：原本這裡在「出發去精定位之前」呼叫 /clear_octomap，導致規劃時 OctoMap
 是空的——植株不在地圖裡（去觀測點、夾完回觀測點等於沒避障），而且「清空後的空
 octree」會讓 MoveIt 碰撞檢查慢到每次規劃 7～16 秒、常常逾時。改成 fine_node 在
 「送新點雲的前一刻」才清空、清完馬上送，手臂規劃時地圖一直都有上一次掃描的植株。
============================================================================
"""
import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor           # 一邊動、一邊聽 YOLO
from rclpy.callback_groups import ReentrantCallbackGroup    # 允許回呼並行，避免卡死

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
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
        # 粗定位（coarse_node.py）選定的目標番茄座標。★ 2026-09-29 起只在「等粗定位」
        # 狀態（_waiting_for_coarse_target=True，/robot_status=COARSE）才收，收到第一筆
        # 就鎖定成這一輪的目標，其餘時間收到的一律忽略——同一輪裡的精定位、備用視角、
        # 夾完回同一個觀測點重掃，繞的都是這一顆，不會被後續訊息換掉。
        self.coarse_target_sub = self.create_subscription(
            PointStamped, '/coarse_target_point', self._coarse_target_callback, 10,
            callback_group=self.cb_group)
        self.latest_coarse_target = None
        # 確定回到 Home 之後（_confirm_home_then_coarse）設成 True：/robot_status 改發
        # COARSE 讓 coarse_node 開始偵測，手臂留在原地等，收到座標才出發。
        self._waiting_for_coarse_target = False
        self._coarse_lock = threading.Lock()
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10, callback_group=self.cb_group)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self._joint_pos_map   = {name: 0.0 for name in config.ARM_JOINT_NAMES}
        self.is_moving        = True
        self.current_step     = 'INIT'
        self.pause_timer      = None
        self.grasp_target     = None
        self.approach_target  = None
        self._return_home_retry_count = 0  # 回初始姿態失敗時的重試次數，成功或開始新一輪失敗處理時歸零
        self.scanning          = False
        self._alt_pose_idx    = 0  # 下一個要試的遮擋備用視角索引，成功找到目標後歸零
        # ★ 這一輪精定位鎖定的番茄座標快照：_move_to_fine 開一輪新的才重新抓
        # self.latest_coarse_target，_move_to_fine_alt 是同一輪裡的備用視角重試，
        # 要繞著同一個物理點轉，不能各自去抓當下最新的座標（coarse_node 是持續在跑的，
        # 兩次呼叫中間可能已經更新，車子還在動的話會抓到不同的番茄，備用視角就繞錯點了）。
        self._fine_target_xyz = None
        # 這次去精定位觀測點（主視角或備用視角）送給 go_to_pose 的法蘭目標
        # (x,y,z,qx,qy,qz,qw,yaw)，到了之後 _confirm_fine_then_scan 拿實際姿態跟它比。
        self._fine_pose_goal = None

        self.get_logger().info('大腦節點啟動！等待 MoveIt Server 連線...')
        self.startup_timer = self.create_timer(0.5, self._check_startup)
        self._status_timer = self.create_timer(0.5, self._republish_status)

    """/robot_status 三種值，由手臂統一決定哪個視覺節點該工作（2026-09-29）：
      COARSE：確定回到 Home、等粗定位給目標 → coarse_node 偵測，fine_node 暫停
      DONE  ：確定到達精定位觀測點、開始掃描 → fine_node 偵測，coarse_node 暫停
      BUSY  ：其餘（移動中、夾取中、IDLE）   → 兩個都暫停"""
    def _current_status(self):
        if self._waiting_for_coarse_target:
            return 'COARSE'
        return 'DONE' if self.scanning else 'BUSY'

    def _republish_status(self):
        self.status_pub.publish(String(data=self._current_status()))

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

    """只在等粗定位（/robot_status=COARSE）時收：鎖定成這一輪的目標，立刻改發 BUSY
    讓 coarse_node 停下來，然後出發去精定位觀測點。其餘時間收到的一律忽略。"""
    def _coarse_target_callback(self, msg: PointStamped):
        # MultiThreadedExecutor + ReentrantCallbackGroup：coarse_node 在交接期間每幀都在發，
        # 兩筆可能同時進來，檢查+清旗標要原子化，不然會出發兩次。
        with self._coarse_lock:
            if not self._waiting_for_coarse_target:
                return
            self._waiting_for_coarse_target = False
        self.latest_coarse_target = (msg.point.x, msg.point.y, msg.point.z)
        self.status_pub.publish(String(data=self._current_status()))
        self.get_logger().info(
            f'收到並鎖定粗定位目標 ({msg.point.x:.3f},{msg.point.y:.3f},{msg.point.z:.3f})，'
            f'粗定位暫停，前往精定位觀測點...')
        self._move_to_fine()

    @property
    def current_joints(self):
        return [self._joint_pos_map[name] for name in config.ARM_JOINT_NAMES]

    def target_callback(self, msg: PoseStamped):
        if not self.scanning:
            return
        self.scanning = False
        self._alt_pose_idx = 0  # 找到目標了，下一輪重新從主視角開始試
        self._process_target(msg)

    def vision_status_callback(self, msg: String):
        if not self.scanning:
            return

        if msg.data == 'OCCLUDED':
            self.scanning = False
            if config.ENABLE_ALT_VIEW and self._alt_pose_idx < len(config.ALT_VIEW_AZIMUTH_OFFSETS_DEG):
                azimuth_offset_deg = config.ALT_VIEW_AZIMUTH_OFFSETS_DEG[self._alt_pose_idx]
                self._alt_pose_idx += 1
                self.get_logger().info(
                    f'這輪候選全被判定遮擋，換備用視角 {self._alt_pose_idx}/'
                    f'{len(config.ALT_VIEW_AZIMUTH_OFFSETS_DEG)}'
                    f'（方位角偏移 {azimuth_offset_deg:+.0f}°）重新掃描...')
                self._move_to_fine_alt(azimuth_offset_deg)
            else:
                self.get_logger().info('備用視角都試過了，還是被遮擋，回初始位置。')
                self._alt_pose_idx = 0
                self._finish_this_round()
            return

        if msg.data != 'NO_TARGET':
            return
        self.scanning = False
        self._alt_pose_idx = 0
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

        # ★ 2026-09-29：出發前先確認預備點 A、夾取點到得了（/compute_ik），到不了直接
        # 報錯、留在觀測點重新掃描，不用等 OMPL 規劃 5 秒失敗、回一個看不懂的 99999。
        self.current_step = 'CHECK_REACH'
        self.is_moving = True
        seeds = [self.current_joints,
                 [math.radians(d) for d in config.POSE_FINE_DEG],
                 [math.radians(d) for d in config.POSE_HOME_DEG]]
        self.arm.check_reachable(self.approach_target, seeds, self._on_approach_reach_checked)

    """預備點 A 的 IK 結果：到得了就接著檢查夾取點。夾取點是從 A 走直線下探過去的，
    所以用 A 的解當種子、不加關節約束（直線路徑本來就不走 OMPL）。"""
    def _on_approach_reach_checked(self, ok, joints):
        if not ok:
            self._report_unreachable('預備點 A', self.approach_target)
            return
        seeds = [joints] if joints is not None else [self.current_joints]
        self.arm.check_reachable(self.grasp_target, seeds, self._on_grasp_reach_checked,
                                 use_joint_constraints=False)

    def _on_grasp_reach_checked(self, ok, _joints):
        if not ok:
            self._report_unreachable('夾取點', self.grasp_target)
            return
        self.current_step = 'APPROACH'
        self.arm.control_gripper(config.GRIPPER_PREOPEN)
        self.get_logger().info('[步驟 1] 預備點 A、夾取點都確認到得了，往預備點 A ...')
        self.arm.go_to_pose(self.approach_target, done_cb=self.on_action_completed)

    """到不了：手臂還沒動、人還在觀測點，直接報錯、放棄這顆，重新開精定位讓使用者改選。"""
    def _report_unreachable(self, name, pose):
        x, y, z = pose[:3]
        self.get_logger().error(
            f'{name}到不了（IK 無解，超出手臂可達範圍或關節約束）：位置=({x:.3f},{y:.3f},{z:.3f})，'
            f'距基座 {math.hypot(x, y, z):.3f}m。放棄這顆，留在觀測點重新掃描。')
        self._enter_scanning()

    def on_action_completed(self, success):
        if not success:
            self.get_logger().warn('動作執行失敗！')
            if self.current_step == 'INIT':
                self._return_home_retry_count += 1
                if self._return_home_retry_count < config.RETURN_HOME_MAX_RETRIES:
                    self.get_logger().warn(
                        f'回初始姿態失敗，重試 {self._return_home_retry_count}/'
                        f'{config.RETURN_HOME_MAX_RETRIES}...')
                    self._move_to_initial()
                else:
                    self.get_logger().error(
                        f'回初始姿態重試 {config.RETURN_HOME_MAX_RETRIES} 次都失敗，'
                        f'停止重試，請人工檢查！')
                    self._reset_to_idle()
            else:
                self.get_logger().warn('嘗試退回初始姿態...')
                self.arm.control_gripper(config.GRIPPER_RELEASE)
                self.current_step = 'INIT'
                self._return_home_retry_count = 0
                self._move_to_initial()
            return

        d = [math.degrees(j) for j in self.current_joints]
        self.get_logger().info(
            f'當前角度: J1={d[0]:.1f}° J2={d[1]:.1f}° J3={d[2]:.1f}° '
            f'J4={d[3]:.1f}° J5={d[4]:.1f}° J6={d[5]:.1f}°')

        step = self.current_step

        if step in ('INIT', 'RETURN'):
            # 啟動、失敗退回、這輪結束回 Home 都走這裡：停頓等手臂穩定 → 確認真的在 Home
            # → 才開粗定位（_confirm_home_then_coarse）。
            self.get_logger().info(f'MoveIt 回報已回到初始位置，停頓 {config.PAUSE_BEFORE_IDLE}s 後確認...')
            self._start_timer(config.PAUSE_BEFORE_IDLE, self._confirm_home_then_coarse)

        elif step == 'TO_FINE':
            self.get_logger().info(f'MoveIt 回報已抵達精定位！停頓 {config.PAUSE_BEFORE_SCAN}s 後確認位置、開始掃描...')
            self._start_timer(config.PAUSE_BEFORE_SCAN, self._confirm_fine_then_scan)

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
                # 回「同一個」精定位觀測點再掃一次，找附近的下一顆；精定位看不到才會
                # 回 Home 重新粗定位（vision_status_callback → _finish_this_round）。
                self.get_logger().info('[步驟 5] 這顆處理完成！回同一個精定位觀測點重新掃描')
                self._move_to_fine()

        elif step == 'BASKET':
            self.get_logger().info('[步驟 7] 抵達籃子上方！放開夾爪')
            self.arm.control_gripper(config.GRIPPER_RELEASE)
            self._start_timer(config.PAUSE_AFTER_RELEASE, self._move_to_fine)

    def _move_to_initial(self):
        self.is_moving = True
        target = [math.radians(deg) for deg in config.POSE_HOME_DEG]
        self.arm.go_to_joints(target, done_cb=self.on_action_completed)

    """手臂相機在 ARM_FLANGE_FRAME(link_6) 座標系下的固定外參：手臂相機不在法蘭軸線上
    （側向偏移＋光軸夾角，見 MathUtils.camera_facing_flange_pose 說明），要拿這個外參
    才能算出「相機對準番茄」對應的法蘭姿態，不能直接拿法蘭姿態當相機姿態用。
    ★ 2026-09-22：改成直接讀 config.CAMERA_EXTRINSIC_TRANSLATION/ROTATION_QUAT
    這組寫死的已知常數，不再查 TF——這條 TF 本來是 fine_node.py 廣播的，只開
    arm_task_node、沒開 fine_node.py 時查不到，會靜默退回「相機=法蘭中心」的簡化假設，
    實測證實這樣算出來的目標點是錯的（法蘭對準了番茄，相機沒有）。這組外參本來就是
    固定常數，不需要透過 TF 才能拿到，直接讀不會受其他節點有沒有開著影響。
    回傳 (cam_t, cam_z)：cam_t 是相機在 link_6 座標系下的位置 (3,)，cam_z 是相機光軸在
    link_6 座標系下的方向 (3,)。"""
    def _get_wrist_camera_extrinsic(self):
        cam_t = np.array(config.CAMERA_EXTRINSIC_TRANSLATION, dtype=float)
        cam_z = R.from_quat(config.CAMERA_EXTRINSIC_ROTATION_QUAT).as_matrix()[:, 2]
        return cam_t, cam_z

    """給番茄 base 座標 (tx,ty,tz)，算出「手臂相機面對番茄、光軸通過番茄中心、離番茄
    COARSE_TO_FINE_RETREAT_M 公尺」時法蘭該去的 (x,y,z,qx,qy,qz,qw,yaw)
    （MathUtils.camera_facing_flange_pose）。
    ★ 2026-09-25：水平接近方向改用 config.FIXED_APPROACH_AZIMUTH_DEG 這個固定角度，
    不再用「基座原點→番茄」算出來的方位角（那個做法會讓每顆番茄的接近方位角跟著
    位置飄動，跟植株排列方向不垂直，畫面看起來歪）。car 場景下，車子每次摘採前都會
    先導航對齊、平行到植株排列方向，同一趟固定往車身側面伸出去摘，方位角本來就該是
    固定值，不是算出來的。azimuth_offset_deg 給 _move_to_fine_alt 的遮擋備用視角用，
    在這個固定角度上再疊加偏移，主流程固定傳 0。"""
    def _camera_facing_pose(self, tx, ty, tz, azimuth_offset_deg=0.0):
        cam_t, cam_z = self._get_wrist_camera_extrinsic()
        az = math.radians(config.FIXED_APPROACH_AZIMUTH_DEG)
        ax, ay = math.cos(az), math.sin(az)
        return MathUtils.camera_facing_flange_pose(
            tx, ty, tz, ax, ay, 0.0, config.COARSE_TO_FINE_RETREAT_M, cam_t, cam_z,
            azimuth_offset_deg=azimuth_offset_deg)

    """目前要接近的番茄座標：這一輪從 coarse_node 鎖定的那一顆（self.latest_coarse_target，
    /coarse_target_point callback 在等粗定位時填的），沒有的話（REQUIRE_COARSE_TARGET=0
    單獨測精定位）才退回 config.FAKE_COARSE_TOMATO_POSE 頂著測，並印警告讓人知道現在是假資料。"""
    def _get_target_tomato_xyz(self):
        if self.latest_coarse_target is not None:
            return self.latest_coarse_target
        if not config.REQUIRE_COARSE_TARGET:
            # 單獨測精定位模式（REQUIRE_COARSE_TARGET=0，不開 coarse_node 也能跑）：
            # 這是預期中會走到的路徑，印一般 warn 就好，不用當成程式漏洞。
            self.get_logger().warn('（單獨測精定位模式）還沒收到 coarse_node 座標，'
                                    '用 FAKE_COARSE_TOMATO_POSE 頂著。')
            return config.FAKE_COARSE_TOMATO_POSE[:3]
        # REQUIRE_COARSE_TARGET=1（預設）時正常流程不會走到這裡——_move_to_fine 只會在
        # _coarse_target_callback 鎖定座標之後、或同一輪夾完回觀測點時被呼叫，
        # latest_coarse_target 一定不是 None。留著純粹當最後一道防線，印大聲一點的警告，
        # 不要讓手臂默默拿假資料動起來卻沒人發現。
        self.get_logger().error('_move_to_fine 在沒有真實座標的情況下被呼叫！暫用 '
                                 'FAKE_COARSE_TOMATO_POSE 頂著，這應該是程式邏輯漏洞，請回報。')
        return config.FAKE_COARSE_TOMATO_POSE[:3]

    """MoveIt 回報回到 Home、停頓完之後呼叫：用 /joint_states 實際量到的關節角確認真的
    在 Home（每軸差距 ≤ HOME_ARRIVAL_TOL_DEG），確認到了才開始新一輪（開粗定位）。
    沒到就當作回 Home 失敗，走 INIT 的重試流程（重試用完轉 IDLE）。"""
    def _confirm_home_then_coarse(self):
        home = [math.radians(d) for d in config.POSE_HOME_DEG]
        diffs = [abs(math.degrees(math.remainder(c - h, 2 * math.pi)))
                 for c, h in zip(self.current_joints, home)]
        worst = max(diffs)
        if worst > config.HOME_ARRIVAL_TOL_DEG:
            self.get_logger().error(
                f'確認 Home 失敗：關節差距最大 {worst:.1f}°（J{diffs.index(worst) + 1}），'
                f'超過 {config.HOME_ARRIVAL_TOL_DEG}°，視為回 Home 失敗。')
            self.current_step = 'INIT'
            self.on_action_completed(False)
            return
        self._return_home_retry_count = 0
        self.get_logger().info(f'確認已在 Home（關節差距最大 {worst:.2f}°）。')
        self.arm.control_gripper(config.GRIPPER_RELEASE)
        self._start_new_round()

    """新一輪：清掉上一輪鎖定的目標，改發 COARSE 讓 coarse_node 開始偵測，手臂留在
    Home 等，收到座標（_coarse_target_callback）才出發。
    ★ config.REQUIRE_COARSE_TARGET=0（單獨測精定位，不開 coarse_node）時不用等，
    直接跑 _move_to_fine()，裡面的 _get_target_tomato_xyz 會自動退回
    FAKE_COARSE_TOMATO_POSE。"""
    def _start_new_round(self):
        self._alt_pose_idx = 0
        self._fine_target_xyz = None
        if not config.REQUIRE_COARSE_TARGET:
            self._move_to_fine()
            return
        self.current_step = 'WAIT_COARSE'
        self.is_moving = False
        self.latest_coarse_target = None
        with self._coarse_lock:
            self._waiting_for_coarse_target = True
        self.status_pub.publish(String(data=self._current_status()))
        self.get_logger().info('開啟粗定位，等待選定目標番茄...')

    """MoveIt 回報抵達精定位觀測點、停頓完之後呼叫：查 TF 拿法蘭實際位姿，跟這次送出的
    目標（_fine_pose_goal）比，位置差 ≤ FINE_ARRIVAL_POS_TOL_M、姿態夾角 ≤
    FINE_ARRIVAL_ORI_TOL_DEG 才開精定位（發 DONE）。查不到 TF 或差太多就當作這一步
    失敗，退回 Home、重新粗定位。"""
    def _confirm_fine_then_scan(self):
        goal = self._fine_pose_goal
        try:
            t = self.tf_buffer.lookup_transform(config.BASE_FRAME, config.EEF_LINK, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error(f'確認精定位失敗：查不到 {config.BASE_FRAME}→{config.EEF_LINK} TF（{e}）。')
            self.on_action_completed(False)
            return
        p, q = t.transform.translation, t.transform.rotation
        pos_err = math.dist((p.x, p.y, p.z), goal[:3])
        dot = abs(float(np.dot(np.array([q.x, q.y, q.z, q.w]), np.array(goal[3:7]))))
        ori_err = math.degrees(2.0 * math.acos(min(1.0, dot)))
        if pos_err > config.FINE_ARRIVAL_POS_TOL_M or ori_err > config.FINE_ARRIVAL_ORI_TOL_DEG:
            self.get_logger().error(
                f'確認精定位失敗：位置差 {pos_err * 1000:.1f}mm（上限 '
                f'{config.FINE_ARRIVAL_POS_TOL_M * 1000:.0f}mm）、姿態差 {ori_err:.1f}°（上限 '
                f'{config.FINE_ARRIVAL_ORI_TOL_DEG}°）。')
            self.on_action_completed(False)
            return
        self.get_logger().info(
            f'確認已在精定位觀測點（位置差 {pos_err * 1000:.1f}mm、姿態差 {ori_err:.1f}°），開啟精定位。')
        self._enter_scanning()

    def _move_to_fine(self):
        """一般精定位流程：走 go_to_pose（座標+姿態）。這裡是每一輪的起點。
        ★ config.REQUIRE_COARSE_TARGET=0（單獨測精定位，不開 coarse_node）時，粗定位
        那條「即時番茄座標 + camera_facing_flange_pose 相機補償」的算法整個跳過（註解
        掉，不執行）——直接用 config.FAKE_COARSE_TOMATO_POSE 整組（8 個數字都用，不是
        只取前 3 個座標）丟給 go_to_pose。這組數字本來就是 2026-09-03 拿 /compute_fk
        對 POSE_FINE_DEG 量出來、再沿姿態的局部 Z 軸往前推 30cm 算出的「假番茄座標」，
        呼叫 MathUtils.retreat_along_local_z 退回 30cm 就精準解回 POSE_FINE_DEG 那個
        已知可達、驗證過的姿勢——這是今天以前本來的行為，單獨測精定位就是要用這條，
        不要被粗定位那條還沒驗證過的新算法影響。
        REQUIRE_COARSE_TARGET=1（預設，正常操作）才會跑下面粗定位那段：把這一輪鎖定的
        目標番茄座標存進 self._fine_target_xyz——之後同一輪如果因為遮擋要換備用視角
        （_move_to_fine_alt），繞的是這個快照。用 _camera_facing_pose 算出「手臂相機
        對準番茄」對應的法蘭姿態。
        會呼叫到這裡的：_coarse_target_callback（新一輪鎖定目標後）、夾完回同一個觀測點
        （LIFT / BASKET 之後）、REQUIRE_COARSE_TARGET=0 時的 _start_new_round。
        ★ 2026-09-30：出發前不再清空 OctoMap，規劃時用上一次掃描的植株避障（見檔頭）。"""
        self.current_step = 'TO_FINE'
        self.is_moving = True

        if not config.REQUIRE_COARSE_TARGET:
            self.get_logger().info('（單獨測精定位模式）直接用 FAKE_COARSE_TOMATO_POSE 退算，'
                                    '不跑粗定位的相機補償算法。')
            cx, cy, cz, cqx, cqy, cqz, cqw, cyaw = config.FAKE_COARSE_TOMATO_POSE
            fx, fy, fz = MathUtils.retreat_along_local_z(cx, cy, cz, cqx, cqy, cqz, cqw,
                                                          config.COARSE_TO_FINE_RETREAT_M)
            self._fine_pose_goal = (fx, fy, fz, cqx, cqy, cqz, cqw, cyaw)
            self.arm.go_to_pose(self._fine_pose_goal, done_cb=self.on_action_completed)
            return

        # ↓↓↓ 粗定位那條路徑：即時番茄座標 + 相機偏移補償（還沒實機/Isaac 驗證過） ↓↓↓
        self._fine_target_xyz = self._get_target_tomato_xyz()
        pose_tuple = self._camera_facing_pose(*self._fine_target_xyz)
        tx, ty, tz = self._fine_target_xyz
        fx, fy, fz, qx, qy, qz, qw, yaw = pose_tuple
        cam_t, cam_z = self._get_wrist_camera_extrinsic()
        r_fl = R.from_quat([qx, qy, qz, qw]).as_matrix()
        cam_pos = np.array([fx, fy, fz]) + r_fl @ cam_t
        cam_dir = r_fl @ (cam_z / np.linalg.norm(cam_z))
        dist_cam_to_target = float(np.linalg.norm(np.array([tx, ty, tz]) - cam_pos))
        self.get_logger().info(
            f'[精定位規劃] 番茄目標=({tx:.3f},{ty:.3f},{tz:.3f}) '
            f'法蘭目標=({fx:.3f},{fy:.3f},{fz:.3f}) yaw={math.degrees(yaw):.1f}° '
            f'推算相機落點=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f}) '
            f'相機到番茄距離={dist_cam_to_target:.3f}m 相機光軸方向={cam_dir.round(3).tolist()}')
        self._fine_pose_goal = pose_tuple
        self.arm.go_to_pose(pose_tuple, done_cb=self.on_action_completed)
        # ↑↑↑ 粗定位那條路徑結束 ↑↑↑

    def _move_to_fine_alt(self, azimuth_offset_deg):
        """遮擋備用視角：繞著跟主流程同一個番茄目標點，把姿態繞世界 Z 軸偏移
        azimuth_offset_deg 度——距離跟「面對目標」都保證不變，只換方位角。
        ★ config.REQUIRE_COARSE_TARGET=0（單獨測精定位）時，跟 _move_to_fine 一樣跳過
        粗定位那條相機補償算法，改用 MathUtils.orbit_around_target 繞著
        FAKE_COARSE_TOMATO_POSE（今天以前本來的行為）。
        REQUIRE_COARSE_TARGET=1 才用 self._fine_target_xyz（這一輪 _move_to_fine 一開始
        鎖定的快照，不是重新抓當下最新座標）+ _camera_facing_pose。

        vision_node 在發 OCCLUDED 之前，已經先清空 OctoMap、用目前這個視角拍了一張
        環境點雲塞進來（見 fine_node.py 的 _publish_environment_mask），這段搖過去的
        路正好會貼近造成遮擋的那叢葉子/藤蔓，靠這份障礙物資料才不會盲搖過去撞上。
        到了新視角之後也不清，等選定目標時 fine_node 才清空、換成新點雲。"""
        self.current_step = 'TO_FINE'
        self.is_moving = True

        if not config.REQUIRE_COARSE_TARGET:
            cx, cy, cz, cqx, cqy, cqz, cqw, _cyaw = config.FAKE_COARSE_TOMATO_POSE
            pose_tuple = MathUtils.orbit_around_target(
                cx, cy, cz, cqx, cqy, cqz, cqw,
                config.COARSE_TO_FINE_RETREAT_M, azimuth_offset_deg)
        else:
            pose_tuple = self._camera_facing_pose(*self._fine_target_xyz,
                                                   azimuth_offset_deg=azimuth_offset_deg)
        self._fine_pose_goal = pose_tuple
        self.arm.go_to_pose(pose_tuple, done_cb=self.on_action_completed)

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

    def _reset_to_idle(self):
        # IDLE 時兩個視覺節點都不該工作（發 BUSY），等人工檢查後重啟。
        self.arm.control_gripper(config.GRIPPER_RELEASE)
        self.is_moving = False
        self.scanning = False
        with self._coarse_lock:
            self._waiting_for_coarse_target = False
        self.current_step = 'IDLE'
        self.status_pub.publish(String(data=self._current_status()))

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
