import rclpy # ROS 2 的 Python 核心庫,建立節點（Node）
import math
import numpy as np # 處理矩陣運算
import threading
import time
from scipy.spatial.transform import Rotation as R # 把「旋轉矩陣」轉成 ROS 用的「四元數」
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor # 多執行緒執行器，讓手臂可以一邊「動」，一邊「聽」YOLO 的訊息
from rclpy.callback_groups import ReentrantCallbackGroup # 允許多個回呼函數同時執行，防止程式因為等手臂動完而卡死

from moveit_msgs.action import MoveGroup, ExecuteTrajectory # MoveGroup 負責找路，ExecuteTrajectory 負責執行路徑
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene # ★ 申請更新場景的服務
# 各種約束條件
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint, BoundingVolume
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject # 碰撞物件與場景訊息、掛載碰撞物件
from shape_msgs.msg import SolidPrimitive # 用來定義虛擬的碰撞物或目標範圍（如：球體、方塊）
from geometry_msgs.msg import Pose, PoseStamped # 描述物體的 3D 位置與姿勢（位置 + 方向）
from sensor_msgs.msg import JointState # 用來控制或讀取關節狀態
#from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

# pos  = Isaac Sim 的 Translate (x, y, z)
# size = Isaac Sim 的 Scale     (x, y, z)

OBSTACLES = [
    {'id': 'table',           'type': 'cube',     'pos': [0.75, -0.1325, 0.03],     'size': [0.7, 1.205, 0.03]},
    {'id': 'front_partition', 'type': 'cube',     'pos': [1.125, -0.1325, 0.29575], 'size': [0.05, 1.205, 0.5015]},
    {'id': 'side_partition',  'type': 'cube',     'pos': [0.499, 0.495, 0.29575],   'size': [1.202, 0.05, 0.5015]},
    {'id': 'computer',        'type': 'cube',     'pos': [0.7, -0.635, 0.25],       'size': [0.46, 0.18, 0.41]},
    {'id': 'box',             'type': 'cube',     'pos': [0.75, 0.0, 0.195],        'size': [0.3, 0.3, 0.3]},
    {'id': 'wall',            'type': 'cube',     'pos': [-0.4, 0.0, 0.5],          'size': [0.06, 2.0, 1.5]},
    {'id': 'basket',          'type': 'cylinder', 'pos': [0.55, -0.63, 0.5063],     'size': [0.1, 0.075]},
    #{'id': 'box',             'type': 'cube',     'pos': [1.0, 0.0, 0.5],        'size': [1.0, 1.0, 1.0]},
]

# 數學與幾何計算工具 (向量幾何大腦)
class MathUtils:
    @staticmethod
    def calculate_grasp_and_approach(tomato_x, tomato_y, tomato_z, gripper_length = 0.156, approach_dist = 0.1):
        # 算出手臂底座要轉幾度才能正對番茄
        yaw = math.atan2(tomato_y, tomato_x)
        
        # 定義夾爪的方向 (Z 軸指目標，Y 軸朝上)
        z_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_axis, z_axis)
        
        # 把軸向組合成旋轉矩陣並轉成四元數 (qx, qy, qz, qw)
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        qx, qy, qz, qw = R.from_matrix(rot_matrix).as_quat()

        # 預留變數存放「抓取點」跟「預備點」
        grasp_x = tomato_x - (gripper_length * z_axis[0])
        grasp_y = tomato_y - (gripper_length * z_axis[1])
        app_x = grasp_x - (approach_dist * z_axis[0])
        app_y = grasp_y - (approach_dist * z_axis[1])

        grasp_target = (grasp_x, grasp_y, tomato_z, qx, qy, qz, qw, yaw)
        approach_target = (app_x, app_y, tomato_z, qx, qy, qz, qw, yaw)
        
        return grasp_target, approach_target

# 手臂與 MoveIt 控制核心 (底層 Wrapper)
class TM5MController:
    def __init__(self, node: Node, cb_group):
        self.node = node
        self.cb_group = cb_group

        # 建立四個窗口：1.找路 2.走路 3.走直線 4.申請更新 Planning Scene 的服務窗口
        self.move_client = ActionClient(node, MoveGroup, 'move_action', callback_group=cb_group)
        self.exec_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=cb_group)
        self.cartesian_client = node.create_client(GetCartesianPath, 'compute_cartesian_path', callback_group=cb_group)
        self.scene_client = node.create_client(ApplyPlanningScene, 'apply_planning_scene', callback_group=cb_group)
        
        # 傳送指令給夾爪
        self.gripper_pub = node.create_publisher(JointState, '/gripper_command', 10)

        # 儲存當前動作完成後的回呼函數，讓不同的動作可以有不同的後續行為
        self._current_done_cb = None  

    def is_ready(self):
        return (self.move_client.server_is_ready() and 
                self.exec_client.server_is_ready() and 
                self.cartesian_client.service_is_ready() and 
                self.scene_client.service_is_ready())

    def control_gripper(self, open_dist):
        msg = JointState()
        msg.name = ['left_finger_joint', 'right_finger_joint']
        msg.position = [float(open_dist), float(open_dist)] 
        self.gripper_pub.publish(msg)
    
    def load_environment(self):#從 OBSTACLES 清單批次建立 Box 碰撞物件，寫入 MoveIt Planning Scene
        success_count = 0
        for obs in OBSTACLES:
            co = CollisionObject()
            co.header.frame_id = 'base'
            co.id = obs['id']
            co.operation = CollisionObject.ADD
            
            pose = Pose()# 設定位置（方向固定朝上，不旋轉）
            pose.position.x, pose.position.y, pose.position.z = float(obs['pos'][0]), float(obs['pos'][1]), float(obs['pos'][2])
            pose.orientation.w = 1.0
            co.primitive_poses = [pose]

            primitive = SolidPrimitive()# 建立方塊幾何體，尺寸直接用 Isaac Sim 的 Scale
            obs_type = obs.get('type', 'box').lower()
            if obs_type == 'cube':
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [float(d) for d in obs['size']]
            elif obs_type == 'cylinder':
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [float(d) for d in obs['size']] # 高度, 半徑
            co.primitives = [primitive]

            # 差異更新：只新增這個物件，不覆蓋整個場景
            req = ApplyPlanningScene.Request()
            req.scene.world.collision_objects.append(co)
            req.scene.is_diff = True
            
            try:
                if self.scene_client.call(req).success:
                    self.node.get_logger().info(f'✓ [{obs["id"]}] 已加入 Planning Scene')
                    success_count += 1
            except Exception as e:
                self.node.get_logger().error(f'載入障礙物 {obs["id"]} 失敗: {e}')
                
        self.node.get_logger().info(f'障礙物載入完成：{success_count}/{len(OBSTACLES)} 個成功')
        
        self._attach_virtual_gripper()# 載入完環境後，順便把虛擬夾爪裝上去
    
    def _attach_virtual_gripper(self):#在法蘭面上動態掛載一個虛擬方塊，代表夾爪，讓 MoveIt 避障時考慮進去
        finger_x = 0.008  
        finger_y = 0.015    
        finger_z = 0.05  
        finger_offset_x = 0.018

        # 建立一個代表夾爪的方塊 (長 15cm, 寬/高 8cm，請依你的夾爪尺寸微調)
        gripper_box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.09, 0.05, 0.10]) # [X寬, Y高, Z長]
        # 設定方塊相對於 flange (法蘭面) 的位置
        gripper_pose = Pose()
        gripper_pose.position.z = 0.05
        gripper_pose.orientation.w = 1.0

        left_finger_box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[finger_x, finger_y, finger_z])
        left_pose = Pose()
        left_pose.position.x = finger_offset_x
        left_pose.position.y = 0.0
        left_pose.position.z = 0.10 + (finger_z / 2.0) # 接在基座 (0.12) 的前端，並置中於手指長度一半
        left_pose.orientation.w = 1.0

        right_finger_box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[finger_x, finger_y, finger_z])
        right_pose = Pose()
        right_pose.position.x = -finger_offset_x
        right_pose.position.y = 0.0
        right_pose.position.z = 0.10 + (finger_z / 2.0) # 同理接在基座前端
        right_pose.orientation.w = 1.0

        # 建立 CollisionObject
        co = CollisionObject(id='virtual_gripper', operation=CollisionObject.ADD)
        co.header.frame_id = 'flange'
        co.primitives, co.primitive_poses = [gripper_box, left_finger_box, right_finger_box], [gripper_pose, left_pose, right_pose]

        # 把 CollisionObject 轉換成「附加」狀態
        aco = AttachedCollisionObject(link_name='flange', object=co)
        
        # 把法蘭面與手腕關節加入白名單，允許夾爪碰到它們 ★★★
        aco.touch_links = ['flange', 'link_6', 'link_5']

        # 發送更新場景的請求
        req = ApplyPlanningScene.Request()
        req.scene.robot_state.attached_collision_objects.append(aco)
        req.scene.robot_state.is_diff = req.scene.is_diff = True

        if self.scene_client.call(req).success:
            self.node.get_logger().info('虛擬夾爪已成功掛載！')
        else:
            self.node.get_logger().warn('虛擬夾爪掛載失敗')
    
    # 動作執行 API
    def go_to_joints(self, target_radians, velocity=0.2, accel=0.2, done_cb=None):
        self._current_done_cb = done_cb
        goal_msg = self._build_joint_goal_msg(target_radians, velocity, accel)
        self._send_action_goal(self.move_client, goal_msg)
        #time.sleep(0.8)

    def go_to_pose(self, pose_tuple, done_cb=None):
        self._current_done_cb = done_cb
        goal_msg = self._build_pose_goal_msg(*pose_tuple)
        self._send_action_goal(self.move_client, goal_msg)

    def execute_cartesian_path(self, target_tuple, done_cb=None):# 笛卡爾直線服務
        self._current_done_cb = done_cb
        x, y, z, qx, qy, qz, qw, _ = target_tuple
        
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base' # 以機器人底座為參考
        req.group_name = 'tmr_arm'
        req.max_step = 0.01 # 直線路徑每 1 公分就算一個點
        req.jump_threshold = 0.0 # 禁用跳躍檢查，防止算不出來
        req.avoid_collisions = True

        target_pose = Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = float(x), float(y), float(z)
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = float(qx), float(qy), float(qz), float(qw)
        req.waypoints.append(target_pose)

        self.node.get_logger().info('啟動純數學直線解算...')
        future = self.cartesian_client.call_async(req) # 請求 MoveIt 算路徑
        future.add_done_callback(self._on_cartesian_planned)

    def _on_cartesian_planned(self, future):
        res = future.result()
        # 如果直線路徑連 90% 都跑不到（可能卡到極限），就報錯停止。
        if res.fraction < 0.95:
            self.node.get_logger().error(f'直線規劃失敗！(可能卡到極限) 完成度: {res.fraction*100:.1f}%')
            #self.node.get_logger().info('啟動復原機制：準備退回初始狀態...')
            #self.node._move_to_initial()
        
            if self._current_done_cb: self._current_done_cb(False)
            return
            
        # 把算好的直線路徑塞進執行指令裡
        goal_msg = ExecuteTrajectory.Goal(trajectory=res.solution)
        self._send_action_goal(self.exec_client, goal_msg)

    # ROS 2 Action 處理生命週期 ---
    def _send_action_goal(self, client, goal_msg):
        future = client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result() # 拿到 Action Server 回傳的處理句柄
        if not goal_handle.accepted:
            if self._current_done_cb: self._current_done_cb(False)
            return
        # 如果接受了，就去「非同步」等待執行結果（有沒有準確走到位）
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_action_result)

    def _on_action_result(self, future):
        # error_code 為 1 代表「成功 (SUCCESS)」
        success = (future.result().result.error_code.val == 1)
        if self._current_done_cb:
            self._current_done_cb(success)

    # MoveIt 訊息建構 
    def _build_joint_goal_msg(self, joint_angles, velocity, accel):
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name, req.pipeline_id, req.planner_id = 'tmr_arm', 'ompl', 'RRTstarkConfigDefault'
        req.allowed_planning_time = 1.5
        req.max_velocity_scaling_factor, req.max_acceleration_scaling_factor = velocity, accel 

        goal_constraints = Constraints()
        for name, angle in zip([f'joint_{i}' for i in range(1, 7)], joint_angles):
            goal_constraints.joint_constraints.append(
                JointConstraint(joint_name=name, position=angle, tolerance_above=0.001, tolerance_below=0.001, weight=1.0))
        req.goal_constraints.append(goal_constraints)
        return goal_msg

    def _build_pose_goal_msg(self, x, y, z, qx, qy, qz, qw, yaw):#OMPL 大範圍移動輔助函數
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name, req.pipeline_id, req.planner_id = 'tmr_arm', 'ompl', 'RRTstarkConfigDefault'
        req.allowed_planning_time, req.num_planning_attempts = 5.0, 15 # 准許它算 3 秒鐘
        req.max_velocity_scaling_factor, req.max_acceleration_scaling_factor = 0.2, 0.2

        target_pose = Pose() # 設定目標位置
        target_pose.position.x, target_pose.position.y, target_pose.position.z = float(x), float(y), float(z)
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = float(qx), float(qy), float(qz), float(qw)

        goal_constraints = Constraints() # 設定約束條件
        
        # 位置約束：告訴手臂 flange (法蘭面) 必須抵達目標點
        pos_con = PositionConstraint(link_name='flange')
        pos_con.header.frame_id = 'base'
        bv = BoundingVolume()
        bv.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001])) # 誤差範圍 0.1 公分
        bv.primitive_poses.append(target_pose)
        pos_con.constraint_region, pos_con.weight = bv, 1.0
        goal_constraints.position_constraints.append(pos_con)

        # 姿勢約束：告訴手臂夾爪的角度要對準
        ori_con = OrientationConstraint(link_name='flange', orientation=target_pose.orientation)
        ori_con.header.frame_id = 'base'
        ori_con.absolute_x_axis_tolerance = ori_con.absolute_y_axis_tolerance = ori_con.absolute_z_axis_tolerance = 0.05
        ori_con.weight = 1.0
        goal_constraints.orientation_constraints.append(ori_con)
        
        # Joint Constraint (Yaw)
        jc1 = JointConstraint(joint_name='joint_1', position = yaw, weight=1.0) # 告訴它：J1 最完美的角度就是直接面對番茄
        jc1.tolerance_above = jc1.tolerance_below = math.radians(40.0) # 給它左右各 60 度的彈性空間閃避奇異點 (絕對不允許轉超過60度繞大圈)
        goal_constraints.joint_constraints.append(jc1)

        #jc5 = JointConstraint(joint_name='joint_5', position = -yaw, weight=1.0) # 告訴它：J5 最完美的角度就是直接面對番茄
        #jc5.tolerance_above = jc5.tolerance_below = math.radians(40.0) # 給它左右各 60 度的彈性空間閃避奇異點 (絕對不允許轉超過60度繞大圈)
        #goal_constraints.joint_constraints.append(jc5)

        req.goal_constraints.append(goal_constraints)
        return goal_msg

# 狀態機與任務排程
class TM5MTaskNode(Node):
    def __init__(self):
        super().__init__('tm5m_task_node')
        self.cb_group = ReentrantCallbackGroup()
        
        # 實例化底層手臂控制器
        self.arm = TM5MController(self, self.cb_group)
        
        # 接收來自 YOLO 的番茄座標
        self.target_sub = self.create_subscription(PoseStamped, '/target_pose', self.target_callback, 10, callback_group=self.cb_group)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        
        self.current_joints = [0.0] * 6
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self._joint_callback, 10,callback_group=self.cb_group)
        
        self.is_moving = True  # 旗標：手臂正在移動時，不接收新指令
        self.current_step = 'INIT' # 狀態機：目前在做哪一步
        self.pause_timer = None # 定時器變數：用來做動作之間的停頓
        self.grasp_target = None   
        self.approach_target = None 

        self.get_logger().info('大腦節點啟動！等待 MoveIt Server 連線...')
        self.startup_timer = self.create_timer(0.5, self._check_startup)

    def _joint_callback(self, msg: JointState):
        self.current_joints = list(msg.position)

    def _check_startup(self):
        if self.arm.is_ready():
            self.startup_timer.cancel()
            self.get_logger().info('所有 Server 已連線！正在載入環境障礙物...')
            self.arm.load_environment()
            self.get_logger().info('正在移動至初始姿態...')
            self._move_to_initial()
        else:
            self.get_logger().info('等待 Server...', throttle_duration_sec=2.0)

    # 視覺觸發 
    def target_callback(self, msg: PoseStamped):
        if self.is_moving: # 如果還在動，不理會新的 YOLO 座標
            return
            
        pos = msg.pose.position
        self.get_logger().info(f'\n番茄中心 X:{pos.x:.3f}, Y:{pos.y:.3f}, Z:{pos.z:.3f}')
        
        # 使用數學工具計算目標點
        self.grasp_target, self.approach_target = MathUtils.calculate_grasp_and_approach(pos.x, pos.y, pos.z)
        
        # 開始執行狀態機：往預備點移動
        self.current_step = 'APPROACH'
        self.is_moving = True
        self.arm.control_gripper(-0.002) 
        self.get_logger().info('[步驟 1] 往預備點 A ...')
        self.arm.go_to_pose(self.approach_target, done_cb=self.on_action_completed)

    # 狀態機切換與定時器邏輯
    def on_action_completed(self, success):# 統一接收所有手臂動作完成的事件，並觸發下一步
        if not success:
            self.get_logger().warn('動作執行失敗！任務重置。')
            self._reset_to_idle()
            return
        
        degs = [math.degrees(j) for j in self.current_joints]
        self.get_logger().info(
            f'當前角度: J1={degs[0]:.1f}° J2={degs[1]:.1f}° '
            f'J3={degs[2]:.1f}° J4={degs[3]:.1f}° '
            f'J5={degs[4]:.1f}° J6={degs[5]:.1f}°')
        
        if self.current_step == 'INIT':# 完成初始定位
            self.get_logger().info('初始定位完成！目前就緒...')
            self._reset_to_idle()
        elif self.current_step == 'APPROACH':# 抵達番茄前的預備點
            self.get_logger().info('抵達點 A，準備直線下探...')
            self._start_timer(2.0, self._step_descend)
        elif self.current_step == 'DESCEND':# 直線前戳到番茄
            self.get_logger().info('[步驟 3] 抵達 Goal！閉合夾爪')
            self.arm.control_gripper(0.015) 
            self._start_timer(1.0, self._step_lift)
        elif self.current_step == 'LIFT':# 夾著番茄退回到預備點
            self.get_logger().info('[步驟 5] 退回點 A 完成！準備前往籃子')
            self._step_to_basket()
        elif self.current_step == 'BASKET':# 抵達籃子上方
            self.get_logger().info('[步驟 7] 抵達籃子上方！放開夾爪')
            self.arm.control_gripper(0.0) 
            self._start_timer(1.0, self._step_return_home)
        elif self.current_step == 'RETURN':# 回到家 (Home)
            self.get_logger().info('完美完成任務！等待下一顆番茄。')
            self._start_timer(1.5, self._final_stabilized_reset)
            #self._reset_to_idle()

    def _final_stabilized_reset(self):
        self.get_logger().info('手臂穩定完畢！完美完成任務，等待下一顆番茄。')
        self._reset_to_idle()  # 這裡面才會把 is_moving 設為 False，並發布 'DONE'

    # 各步驟具體執行函數
    def _move_to_initial(self):
        self.is_moving = True
        target_radians = [math.radians(deg) for deg in [0.0, -10.0, 40.0, 25.0, 90.0, 0.0]]
        self.arm.go_to_joints(target_radians, velocity=0.2, accel=0.2, done_cb = self.on_action_completed)

    def _step_descend(self):
        self.current_step = 'DESCEND'
        self.get_logger().info('[步驟 2] 暫停結束！筆直前戳到 Goal')
        self.arm.execute_cartesian_path(self.grasp_target, done_cb = self.on_action_completed)

    def _step_lift(self):
        self.current_step = 'LIFT'
        self.get_logger().info('[步驟 4] 夾取完成！原路退回點 A')
        self.arm.execute_cartesian_path(self.approach_target, done_cb = self.on_action_completed)

    def _step_to_basket(self):
        self.current_step = 'BASKET'
        self.get_logger().info('[步驟 6] 前往籃子上方...')
        # 這是你從 RViz 截圖中拉出來的完美關節角度 (度數)，轉換為弧度
        target_radians = [math.radians(deg) for deg in [-42.0, 29.0, 31.0, -15.0, 90.0, 0.0]]
        self.arm.go_to_joints(target_radians, velocity=0.2, accel=0.2, done_cb = self.on_action_completed)

    def _step_return_home(self):
        self.current_step = 'RETURN'
        self.get_logger().info('[步驟 8] 回到初始位置')
        self._move_to_initial()

    def _reset_to_idle(self):
        self.arm.control_gripper(0.0)
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
    rclpy.init(args=args) # 初始化 ROS 2 系統
    node = TM5MTaskNode() # 創建你的大腦節點
    executor = MultiThreadedExecutor() # 創建多執行緒引擎
    executor.add_node(node) # 把大腦放進引擎
    try:
        executor.spin() # 讓節點開始跑起來，不斷檢查有沒有訊息要處理
    except KeyboardInterrupt: # 如果你按了 Ctrl+C
        pass
    finally:
        node.destroy_node() # 關閉節點
        rclpy.shutdown()    # 關閉 ROS 2

if __name__ == '__main__':
    main()