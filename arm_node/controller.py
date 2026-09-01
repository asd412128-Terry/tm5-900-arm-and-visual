"""
============================================================================
 arm_node.controller — 手臂控制核心：MoveIt 底層 Wrapper
============================================================================
 職責：封裝 MoveIt 動作 API（joint/pose/cartesian 目標、夾爪控制、
 Action 生命週期），不含任務流程邏輯（那是 arm_task_node.py 的事）。
============================================================================
"""
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene, GetStateValidity
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             JointConstraint, BoundingVolume, RobotState)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty  # ★ 用來觸發 octomap 清空重建（精定位前抓一次點雲快照）
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
        self.clear_octomap_client = node.create_client(Empty, '/clear_octomap', callback_group=cb_group)
        self.set_io_client = node.create_client(SetIO, '/set_io', callback_group=cb_group)

        self.gripper_pub = node.create_publisher(JointState, '/gripper_command', 10)
        self.gripper_state_pub = node.create_publisher(JointState, '/joint_states', 10)
        self.scene = SceneBuilder(node, self.scene_client)

        self._current_done_cb = None
        self._cartesian_vel_scale = config.CART_VEL

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
        self._send_action_goal(self.move_client,
                               self._build_joint_goal_msg(target_radians, velocity, accel))

    def go_to_pose(self, pose_tuple, done_cb=None):
        self._current_done_cb = done_cb
        self._send_action_goal(self.move_client, self._build_pose_goal_msg(*pose_tuple))

    def execute_cartesian_path(self, target_tuple, done_cb=None, velocity=config.CART_VEL, accel=config.CART_ACC):
        self._current_done_cb = done_cb
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
            if self._current_done_cb:
                self._current_done_cb(False)
            return

        slowed = self._retime_trajectory(res.solution, self._cartesian_vel_scale)
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
        client.send_goal_async(goal_msg).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self._current_done_cb:
                self._current_done_cb(False)
            return
        goal_handle.get_result_async().add_done_callback(self._on_action_result)

    def _on_action_result(self, future):
        result = future.result().result
        success = (result.error_code.val == 1)
        if not success:
            self.log.error(f'MoveIt error_code: {result.error_code.val}')
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

        # (c) J1 約束
        jc1 = JointConstraint(joint_name='joint_1', position=yaw, weight=1.0)
        jc1.tolerance_above = jc1.tolerance_below = config.J1_TOLERANCE
        gc.joint_constraints.append(jc1)

        # (d) 手肘 (joint_3) 約束：鎖在「手肘朝上」那個分支附近，避免 OMPL 選到手肘
        # 翻到另一側的替代解。
        jc3 = JointConstraint(joint_name='joint_3', position=config.ELBOW_UP_CENTER, weight=1.0)
        jc3.tolerance_above = jc3.tolerance_below = config.ELBOW_UP_TOLERANCE
        gc.joint_constraints.append(jc3)

        req.goal_constraints.append(gc)
        return goal_msg
