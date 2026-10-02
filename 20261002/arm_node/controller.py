"""
============================================================================
 arm_node.controller — 手臂控制核心：MoveIt 底層 Wrapper
============================================================================
 職責：封裝 MoveIt 動作 API（joint/pose/cartesian 目標、夾爪控制、
 Action 生命週期），不含任務流程邏輯（那是 arm_task_node.py 的事）。
============================================================================
"""
import math
import time

from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene, GetStateValidity, GetPositionIK
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             JointConstraint, BoundingVolume, RobotState)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from tm_msgs.srv import SetIO

from . import config
from .scene_builder import SceneBuilder


class TM5MController:
    def __init__(self, node: Node, cb_group):
        self.node = node
        self.log = node.get_logger()

        self.move_client      = ActionClient(node, MoveGroup, 'move_action', callback_group=cb_group)
        self.exec_client      = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=cb_group)
        self.cartesian_client = node.create_client(GetCartesianPath, 'compute_cartesian_path', callback_group=cb_group)
        self.scene_client     = node.create_client(ApplyPlanningScene, 'apply_planning_scene', callback_group=cb_group)
        self.state_validity_client = node.create_client(GetStateValidity, 'check_state_validity', callback_group=cb_group)
        self.ik_client = node.create_client(GetPositionIK, 'compute_ik', callback_group=cb_group)
        self.set_io_client = node.create_client(SetIO, '/set_io', callback_group=cb_group)

        self.gripper_pub = node.create_publisher(JointState, '/gripper_command', 10)
        self.gripper_state_pub = node.create_publisher(JointState, '/joint_states', 10)
        self.scene = SceneBuilder(node, self.scene_client)

        self._current_done_cb = None
        self._cartesian_vel_scale = config.CART_VEL

        # 2026-09-30：每次移動的計時（見 _start_motion_timer / _report_motion_time）。
        # 用牆上時間（time.monotonic），反映實際等了多久，不受 Isaac 模擬時鐘快慢影響。
        self._motion_kind = ''
        self._motion_t0 = None        # 送出請求的時間
        self._motion_exec_t0 = None   # 規劃完、開始移動的時間（None = 還沒開始移動）

        self._last_gripper_pos = config.GRIPPER_RELEASE
        self._gripper_state_timer = node.create_timer(
            0.5, self._republish_gripper_state, callback_group=cb_group)
        self._republish_gripper_state()

    def _republish_gripper_state(self):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ['left_finger_joint', 'right_finger_joint']
        msg.position = [float(self._last_gripper_pos)] * 2
        self.gripper_state_pub.publish(msg)

    def is_ready(self):
        return (self.move_client.server_is_ready() and
                self.exec_client.server_is_ready() and
                self.cartesian_client.service_is_ready() and
                self.scene_client.service_is_ready())

    def check_current_state_validity(self, joint_names, joint_positions, group_name=None):
        if not self.state_validity_client.service_is_ready():
            self.log.warn('check_state_validity service 尚未就緒，跳過診斷')
            return

        req = GetStateValidity.Request()
        req.group_name = config.ARM_GROUP if group_name is None else group_name
        rs = RobotState()
        js = JointState()
        js.name = list(joint_names)
        js.position = list(joint_positions)
        rs.joint_state = js
        req.robot_state = rs

        future = self.state_validity_client.call_async(req)

        def _on_result(fut):
            res = fut.result()
            if res.valid:
                self.log.info(f'✓ 診斷[{req.group_name or "ALL"}]：目前狀態合法，沒有碰撞')
                return
            self.log.error(f'✗ 診斷[{req.group_name or "ALL"}]：目前狀態非法！共 {len(res.contacts)} 組碰撞：')
            for c in res.contacts:
                self.log.error(f'    {c.contact_body_1} <-> {c.contact_body_2}')

        future.add_done_callback(_on_result)

    def load_environment(self):
        self.scene.build_all()

    def control_gripper(self, open_dist):
        self._last_gripper_pos = float(open_dist)
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ['left_finger_joint', 'right_finger_joint']
        msg.position = [float(open_dist), float(open_dist)]
        self.gripper_pub.publish(msg)
        self.gripper_state_pub.publish(msg)

        is_close = (float(open_dist) == config.GRIPPER_GRASP)
        self._set_real_gripper(close=is_close)

    def _set_real_gripper(self, close: bool):
        if config.ARM_MODE == 'isaac':
            return
        if not self.set_io_client.wait_for_service(timeout_sec=1.0):
            self.log.warn('/set_io service 等待逾時，跳過實體夾爪控制')
            return
        req = SetIO.Request()
        req.module = config.GRIPPER_IO_MODULE
        req.type = config.GRIPPER_IO_TYPE
        req.pin = config.GRIPPER_IO_PIN
        req.state = (config.GRIPPER_IO_CLOSE_STATE if close
                     else config.GRIPPER_IO_OPEN_STATE)
        self.set_io_client.call_async(req)

    def go_to_joints(self, target_radians, velocity=config.JOINT_VEL, accel=config.JOINT_ACC, done_cb=None):
        self._current_done_cb = done_cb
        self._start_motion_timer('關節移動')
        self._send_action_goal(self.move_client,
                               self._build_joint_goal_msg(target_radians, velocity, accel))

    def go_to_pose(self, pose_tuple, done_cb=None):
        self._current_done_cb = done_cb
        self._start_motion_timer('位姿移動')
        self._send_action_goal(self.move_client, self._build_pose_goal_msg(*pose_tuple))

    def _start_motion_timer(self, kind):
        self._motion_kind = kind
        self._motion_t0 = time.monotonic()
        self._motion_exec_t0 = None

    def _mark_motion_executing(self):
        if self._motion_exec_t0 is None:
            self._motion_exec_t0 = time.monotonic()

    """MoveGroup action 的 feedback：state 從 PLANNING 變成 MONITOR 就代表規劃完、開始移動。"""
    def _on_move_feedback(self, feedback_msg):
        if feedback_msg.feedback.state == 'MONITOR':
            self._mark_motion_executing()

    """每次移動結束（成功或失敗）印一行：總時間 = 規劃 + 移動。
    規劃階段就失敗（沒開始移動）時只有規劃時間。步驟名稱取自 arm_task_node 的 current_step。"""
    def _report_motion_time(self, success):
        if self._motion_t0 is None:
            return
        now = time.monotonic()
        step = getattr(self.node, 'current_step', '')
        result = '成功' if success else '失敗'
        total = now - self._motion_t0
        if self._motion_exec_t0 is not None:
            plan = self._motion_exec_t0 - self._motion_t0
            move = now - self._motion_exec_t0
            detail = f'規劃 {plan:.2f}s + 移動 {move:.2f}s'
        else:
            detail = f'規劃 {total:.2f}s，沒有開始移動'
        msg = f'[計時] {step} {self._motion_kind}{result}：總共 {total:.2f}s（{detail}）'
        # rclpy 的 logger 同一個呼叫位置不能換等級（會丟 ValueError），成功/失敗要分開兩行呼叫。
        if success:
            self.log.info(msg)
        else:
            self.log.warn(msg)
        self._motion_t0 = None

    """出發前檢查一個法蘭位姿到不到得了（/compute_ik，只算運動學、不檢查碰撞）。
    依序拿 seeds 裡的關節角當 IK 種子（KDL 的隨機重啟容易漏掉窄區域的解，多給幾個
    已知好的起點比較不會誤判），任一個解出來就算到得了。
    use_joint_constraints=True 時帶上跟 go_to_pose 同一組 J1/J3/J6/J5 約束。
    結果非同步回呼 done_cb(ok, joints)：ok=False 時 joints 是 None。"""
    def check_reachable(self, pose_tuple, seeds, done_cb, use_joint_constraints=True):
        if not self.ik_client.service_is_ready():
            self.log.warn('compute_ik service 尚未就緒，跳過可達性檢查，視為到得了。')
            done_cb(True, None)
            return
        x, y, z, qx, qy, qz, qw = pose_tuple[:7]
        constraints = Constraints()
        if use_joint_constraints:
            constraints.joint_constraints.extend(self._pose_joint_constraints(x, y))
        self._try_ik_seed(self._make_pose(x, y, z, qx, qy, qz, qw), constraints, list(seeds), done_cb)

    def _try_ik_seed(self, pose, constraints, seeds, done_cb):
        if not seeds:
            done_cb(False, None)
            return
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = config.ARM_GROUP
        ik.ik_link_name = config.EEF_LINK
        ik.avoid_collisions = False
        ik.robot_state.joint_state.name = list(config.ARM_JOINT_NAMES)
        ik.robot_state.joint_state.position = [float(a) for a in seeds[0]]
        ik.pose_stamped = PoseStamped(pose=pose)
        ik.pose_stamped.header.frame_id = config.BASE_FRAME
        ik.constraints = constraints
        ik.timeout.nanosec = int(config.REACH_IK_TIMEOUT_SEC * 1e9)

        def _on_result(future):
            res = future.result()
            if res is not None and res.error_code.val == 1:
                sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                done_cb(True, [sol[name] for name in config.ARM_JOINT_NAMES])
            else:
                self._try_ik_seed(pose, constraints, seeds[1:], done_cb)

        self.ik_client.call_async(req).add_done_callback(_on_result)

    def execute_cartesian_path(self, target_tuple, done_cb=None, velocity=config.CART_VEL, accel=config.CART_ACC):
        self._current_done_cb = done_cb
        self._start_motion_timer('直線移動')
        x, y, z, qx, qy, qz, qw, _ = target_tuple

        req = GetCartesianPath.Request()
        req.header.frame_id = config.BASE_FRAME
        req.group_name = config.ARM_GROUP
        req.max_step = config.CART_MAX_STEP
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        self._cartesian_vel_scale = max(1e-3, min(float(velocity), 1.0))
        req.waypoints.append(self._make_pose(x, y, z, qx, qy, qz, qw))

        self.log.info('啟動純數學直線解算...')
        self.cartesian_client.call_async(req).add_done_callback(self._on_cartesian_planned)

    def _on_cartesian_planned(self, future):
        res = future.result()
        if res.fraction < config.CART_MIN_FRACTION:
            self.log.error(f'直線規劃失敗！完成度: {res.fraction * 100:.1f}%')
            self._report_motion_time(False)
            if self._current_done_cb:
                self._current_done_cb(False)
            return

        slowed = self._retime_trajectory(res.solution, self._cartesian_vel_scale)
        self._mark_motion_executing()   # 直線路徑算完就開始執行
        self._send_action_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=slowed))

    @staticmethod
    def _retime_trajectory(robot_traj, scale):
        if scale >= 0.999:
            return robot_traj
        inv = 1.0 / scale
        for pt in robot_traj.joint_trajectory.points:
            total_ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            total_ns = int(total_ns * inv)
            pt.time_from_start.sec = total_ns // 1_000_000_000
            pt.time_from_start.nanosec = total_ns % 1_000_000_000
            if pt.velocities:
                pt.velocities = [v * scale for v in pt.velocities]
            if pt.accelerations:
                pt.accelerations = [a * scale * scale for a in pt.accelerations]
        return robot_traj

    def _send_action_goal(self, client, goal_msg):
        feedback_cb = self._on_move_feedback if client is self.move_client else None
        client.send_goal_async(goal_msg, feedback_callback=feedback_cb).add_done_callback(
            self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._report_motion_time(False)
            if self._current_done_cb:
                self._current_done_cb(False)
            return
        goal_handle.get_result_async().add_done_callback(self._on_action_result)

    def _on_action_result(self, future):
        result = future.result().result
        success = (result.error_code.val == 1)
        if not success:
            self.log.error(f'MoveIt error_code: {result.error_code.val}')
        self._report_motion_time(success)
        if self._current_done_cb:
            self._current_done_cb(success)

    @staticmethod
    def _make_pose(x, y, z, qx, qy, qz, qw) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.x, p.orientation.y = float(qx), float(qy)
        p.orientation.z, p.orientation.w = float(qz), float(qw)
        return p

    def _build_joint_goal_msg(self, joint_angles, velocity, accel):
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name, req.pipeline_id, req.planner_id = config.ARM_GROUP, config.PIPELINE_ID, config.PLANNER_ID
        req.allowed_planning_time = config.PLAN_TIME_JOINT
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = accel

        gc = Constraints()
        for name, angle in zip([f'joint_{i}' for i in range(1, 7)], joint_angles):
            gc.joint_constraints.append(
                JointConstraint(joint_name=name, position=angle,
                                tolerance_above=0.001, tolerance_below=0.001, weight=1.0))
        req.goal_constraints.append(gc)
        return goal_msg

    def _build_pose_goal_msg(self, x, y, z, qx, qy, qz, qw, yaw):
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name, req.pipeline_id, req.planner_id = config.ARM_GROUP, config.PIPELINE_ID, config.PLANNER_ID
        req.allowed_planning_time = config.PLAN_TIME_POSE
        req.num_planning_attempts = config.PLAN_ATTEMPTS
        req.max_velocity_scaling_factor = config.POSE_VEL
        req.max_acceleration_scaling_factor = config.POSE_ACC

        target_pose = self._make_pose(x, y, z, qx, qy, qz, qw)
        gc = Constraints()

        # (a) 位置約束
        pos_con = PositionConstraint(link_name=config.EEF_LINK)
        pos_con.header.frame_id = config.BASE_FRAME
        bv = BoundingVolume()
        bv.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[config.POS_TOLERANCE]))
        bv.primitive_poses.append(target_pose)
        pos_con.constraint_region, pos_con.weight = bv, 1.0
        gc.position_constraints.append(pos_con)

        # (b) 姿態約束
        ori_con = OrientationConstraint(link_name=config.EEF_LINK, orientation=target_pose.orientation)
        ori_con.header.frame_id = config.BASE_FRAME
        ori_con.absolute_x_axis_tolerance = config.ORI_TOLERANCE
        ori_con.absolute_y_axis_tolerance = config.ORI_TOLERANCE
        ori_con.absolute_z_axis_tolerance = config.ORI_TOLERANCE
        ori_con.weight = 1.0
        gc.orientation_constraints.append(ori_con)

        # (c)~(f) J1/J3/J6/J5 關節約束：抽到 _pose_joint_constraints，出發前的 IK 可達性
        # 檢查（check_reachable）也用同一組，兩邊判斷才會一致。
        gc.joint_constraints.extend(self._pose_joint_constraints(x, y))

        req.goal_constraints.append(gc)
        return goal_msg

    def _pose_joint_constraints(self, x, y):
        """go_to_pose 目標額外加的 J1/J6/J5 關節約束（J3 目前註解掉）。
        (x, y) 是這一步法蘭要到的點（base 座標），J1 中心 = 基座→這個點的方位角。"""
        constraints = []

        # (c) J1 約束
        # ★ 2026-09-30：中心改回 yaw，而且統一用「基座 → 這一步要到的點」的方位角
        # atan2(y, x)，直接用傳進來的目標點算，不依賴上游各自算的 yaw——觀測點、備用視角、
        # 預備點 A 全部同一套。09-23 以前的 yaw 版本失敗，是因為預備點 A 用的是果梗點的
        # 方位角（不跟著備用視角移動），走 −50° 視角時需要的 J1 超出 ±40°；改用要到的點
        # 之後，實測差距 1～27°（手臂側向偏移造成），±40° 內。
        # 09-23～09-30 用的「出發前當下角度」版本留著、註解掉：
        # current_j1 = self.node.current_joints[0]
        # jc1 = JointConstraint(joint_name='joint_1', position=current_j1, weight=1.0)
        yaw = math.atan2(y, x)
        jc1 = JointConstraint(joint_name='joint_1', position=yaw, weight=1.0)
        jc1.tolerance_above = jc1.tolerance_below = config.J1_TOLERANCE
        constraints.append(jc1)

        # (d) 手肘 (joint_3) 約束：鎖在「手肘朝上」那個分支附近，避免 OMPL 選到手肘
        # 翻到另一側的替代解。
        # ★ 2026-09-23 實驗性改法：中心同樣改成「出發前 joint_3 當下的角度」，不再是
        # 固定的 ELBOW_UP_CENTER=90°。原本的寫法留著、註解掉：
        # jc3 = JointConstraint(joint_name='joint_3', position=config.ELBOW_UP_CENTER, weight=1.0)
        # ★ 2026-09-30 實驗：J3 完全不加約束（只受硬體限位），看規劃失敗會不會減少。
        # 要恢復就把下面四行取消註解。
        # current_j3 = self.node.current_joints[2]
        # jc3 = JointConstraint(joint_name='joint_3', position=current_j3, weight=1.0)
        # jc3.tolerance_above = jc3.tolerance_below = config.ELBOW_UP_TOLERANCE
        # constraints.append(jc3)

        # (e) J6 約束：鎖在 0 度附近 ±90 度，避免 OMPL 選到要多轉一整圈才能到的等效解。
        # ★ 2026-09-23：曾經試過改成「當下角度」（跟 J1/J3 一樣），但 J6 跟 J1/J3 不
        # 一樣——J6 控制腕部自轉，直接決定掛在軸線外的東西（例如手臂相機）現在是朝上
        # 還是朝下；改成當下角度會失去「拉回接近 0 度」這個絕對錨點，連續移動好幾次
        # 之後 J6 可能一路飄到接近 180 度，相機因此整個上下顛倒（實機截圖證實過）。
        # 改回固定版本，避免這個視覺上會出問題的漂移。曾經試過的當下角度版本：
        # current_j6 = self.node.current_joints[5]
        # jc6 = JointConstraint(joint_name='joint_6', position=current_j6, weight=1.0)
        jc6 = JointConstraint(joint_name='joint_6', position=config.J6_CENTER, weight=1.0)
        jc6.tolerance_above = jc6.tolerance_below = config.J6_TOLERANCE
        constraints.append(jc6)

        # (f) J5 約束：鎖在固定 J5_CENTER=90° 附近 ±90°（[0°, 180°]），不跨過 0° 的
        # 腕部奇異點、擋住腕部翻轉。
        # ★ 2026-10-01：原本中心是「出發前 joint_5 當下的角度」±180°，備用視角之間
        # J5 一次要轉到 175°（見 config.py J5_CENTER 的說明），改成固定中心。原本的寫法：
        # current_j5 = self.node.current_joints[4]
        # jc5 = JointConstraint(joint_name='joint_5', position=current_j5, weight=1.0)
        # jc5.tolerance_above = jc5.tolerance_below = config.J5_MAX_STEP
        jc5 = JointConstraint(joint_name='joint_5', position=config.J5_CENTER, weight=1.0)
        jc5.tolerance_above = jc5.tolerance_below = config.J5_TOLERANCE
        constraints.append(jc5)
        return constraints
