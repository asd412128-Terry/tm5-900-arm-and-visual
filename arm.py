import rclpy # ROS 2 的 Python 核心庫,建立節點（Node）
import math
import numpy as np # 處理矩陣運算
from scipy.spatial.transform import Rotation as R #把「旋轉矩陣」轉成 ROS 用的「四元數」
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor # 多執行緒執行器，讓手臂可以一邊「動」，一邊「聽」YOLO 的訊息
from rclpy.callback_groups import ReentrantCallbackGroup # 允許多個回呼函數同時執行，防止程式因為等手臂動完而卡死

from moveit_msgs.action import MoveGroup, ExecuteTrajectory # MoveGroup 負責找路，ExecuteTrajectory 負責執行路徑
from moveit_msgs.srv import GetCartesianPath # 直線路徑規劃服務
# 各種約束條件
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive # 用來定義虛擬的碰撞物或目標範圍（如：球體、方塊
from geometry_msgs.msg import Pose, PoseStamped  # 描述物體的 3D 位置與姿勢（位置 + 方向）
from sensor_msgs.msg import JointState # 用來控制或讀取關節狀態
from std_msgs.msg import String

class TM5MControlNode(Node):
    def __init__(self):
        super().__init__('tm5m_control_node')
        
        self.cb_group = ReentrantCallbackGroup()
        # 建立三個窗口：1. 找路 2. 走路 3. 走直線
        self._action_client = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cb_group)
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory', callback_group=self.cb_group)
        self.cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path', callback_group=self.cb_group)
        
        # 傳送指令給夾爪
        self.gripper_pub = self.create_publisher(JointState, '/gripper_command', 10)
        # 接收來自 YOLO 的番茄座標
        self.subscription = self.create_subscription(PoseStamped, '/target_pose', self.target_callback, 10, callback_group=self.cb_group)
        
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.is_moving = True  # 旗標：手臂正在移動時，不接收新指令
        self.current_step = 'INIT' # 狀態機：目前在做哪一步
        self.pause_timer = None # 定時器變數：用來做動作之間的停頓
        # 預留變數存放「抓取點」跟「預備點」
        self.grasp_target = None   
        self.approach_target = None 

        self.get_logger().info('機器手臂肌肉節點起動！等待所有 MoveIt Server 連線...')
        self.startup_timer = self.create_timer(0.5, self.startup_timer_callback)

    def startup_timer_callback(self):
        if self._action_client.server_is_ready() and self.execute_client.server_is_ready() and self.cartesian_client.service_is_ready():
            self.startup_timer.cancel()
            self.get_logger().info('所有 Server 已連線！正在移動至初始姿態...')
            self.go_to_initial_joints(velocity=0.05, accel=0.05) 
        else:
            self.get_logger().info('等待 Server...', throttle_duration_sec=2.0)

    def control_gripper(self, open_dist):
        msg = JointState()
        msg.name = ['left_finger_joint', 'right_finger_joint']
        msg.position = [float(open_dist), float(open_dist)] 
        self.gripper_pub.publish(msg)

    # ==========================================================
    # 向量幾何大腦 (保持與影片一樣的完美姿勢)
    # ==========================================================
    def target_callback(self, msg: PoseStamped):
        if self.is_moving: # 如果還在動，不理會新的 YOLO 座標
            return
            
        tomato_x = msg.pose.position.x 
        tomato_y = msg.pose.position.y
        tomato_z = msg.pose.position.z
        self.get_logger().info(f'\n番茄中心 X:{tomato_x:.3f}, Y:{tomato_y:.3f}, Z:{tomato_z:.3f}')
        # 算出手臂底座要轉幾度才能正對番茄
        yaw = math.atan2(tomato_y, tomato_x)
        # 定義夾爪的方向 (Z 軸指目標，Y 軸朝上)
        z_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_axis, z_axis)
        # 把軸向組合成旋轉矩陣並轉成四元數 (qx, qy, qz, qw)
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        qx, qy, qz, qw = R.from_matrix(rot_matrix).as_quat()

        gripper_length = 0.106 
        approach_dist = 0.1   

        grasp_x = tomato_x - (gripper_length * z_axis[0])
        grasp_y = tomato_y - (gripper_length * z_axis[1])

        app_x = grasp_x - (approach_dist * z_axis[0])
        app_y = grasp_y - (approach_dist * z_axis[1])

        self.grasp_target = (grasp_x, grasp_y, tomato_z, qx, qy, qz, qw, yaw)
        self.approach_target = (app_x, app_y, tomato_z, qx, qy, qz, qw, yaw)
        # 開始執行狀態機：往預備點移動
        self.current_step = 'APPROACH'
        self.control_gripper(0.0) 
        self.get_logger().info(f'[步驟 1]往預備點 A ...')
        self.go_to_pose(*self.approach_target)

    # ==========================================================
    # 笛卡爾直線服務
    # ==========================================================
    def execute_cartesian_path(self, target_tuple):
        try:
            self.is_moving = True 
            x, y, z, qx, qy, qz, qw, yaw = target_tuple
            
            req = GetCartesianPath.Request()
            req.header.frame_id = 'base' # 以機器人底座為參考
            req.group_name = 'tmr_arm'
            
            req.max_step = 0.01 # 直線路徑每 1 公分就算一個點
            req.jump_threshold = 0.0 # 禁用跳躍檢查，防止算不出來

            target_pose = Pose()
            target_pose.position.x = float(x)
            target_pose.position.y = float(y)
            target_pose.position.z = float(z)
            target_pose.orientation.x = float(qx)
            target_pose.orientation.y = float(qy)
            target_pose.orientation.z = float(qz)
            target_pose.orientation.w = float(qw)
            req.waypoints.append(target_pose)
            
            self.get_logger().info('啟動純數學直線解算...')
            future = self.cartesian_client.call_async(req) # 請求 MoveIt 算路徑
            future.add_done_callback(self.cartesian_plan_callback)
            
        except Exception as e:
            self.get_logger().error(f'構造直線規劃請求時發生嚴重錯誤: {e}')
            self.is_moving = False
            self.current_step = 'IDLE'

    def cartesian_plan_callback(self, future):
        try:
            res = future.result()
            if res.fraction < 0.9: # 如果直線路徑連 90% 都跑不到（可能卡到極限），就報錯停止。
                self.get_logger().error(f'直線規劃失敗！(可能卡到極限) 完成度: {res.fraction*100:.1f}%')
                self.is_moving = False
                self.current_step = 'IDLE'
                return
            
            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = res.solution # 把算好的直線路徑塞進執行指令裡
            self._send_exec_future = self.execute_client.send_goal_async(goal_msg)
            self._send_exec_future.add_done_callback(self.exec_response_callback)
            
        except Exception as e:
            self.get_logger().error(f'Cartesian service 回呼發生錯誤: {e}')
            self.is_moving = False
            self.current_step = 'IDLE'

    def exec_response_callback(self, future):
        goal_handle = future.result() # 拿到 Action Server 回傳的處理句柄
        if not goal_handle.accepted:
            self.is_moving = False
            self.current_step = 'IDLE'
            return
        # 如果接受了，就去「非同步」等待執行結果（有沒有準確走到位）
        self._get_exec_result_future = goal_handle.get_result_async()
        self._get_exec_result_future.add_done_callback(self.get_result_callback)

    # ==========================================================
    # 狀態機切換邏輯
    # ==========================================================
    def continue_to_descend(self):
        if self.pause_timer:
            self.pause_timer.cancel()
        self.current_step = 'DESCEND'
        self.get_logger().info('[步驟 2] 暫停結束！筆直前戳到 Goal')
        self.execute_cartesian_path(self.grasp_target)

    def grasp_timer_callback(self):
        if self.pause_timer:
            self.pause_timer.cancel()
        self.current_step = 'LIFT'
        self.get_logger().info('[步驟 4] 夾取完成！原路完美退回點 A')
        self.execute_cartesian_path(self.approach_target)

    def get_result_callback(self, future):
        result = future.result().result # 拿到動作執行的最終結果
        msg = String()
        msg.data = 'DONE'
        if result.error_code.val == 1: # error_code 為 1 代表「成功 (SUCCESS)」
            # 剛完成初始定位 
            if self.current_step == 'INIT':
                self.get_logger().info('初始定位完成！目前就緒...')
                self.control_gripper(0.0) 
                self.is_moving = False
                self.current_step = 'IDLE'
                self.status_pub.publish(msg)

            # 情況 B：剛抵達番茄前的預備點
            elif self.current_step == 'APPROACH':
                self.get_logger().info('抵達點 A...')
                self.pause_timer = self.create_timer(2.0, self.continue_to_descend)
            # 情況 C：剛直線前戳到番茄     
            elif self.current_step == 'DESCEND':
                self.get_logger().info('[步驟 3] 抵達 Goal！閉合夾爪')
                self.control_gripper(0.015) 
                self.pause_timer = self.create_timer(1.0, self.grasp_timer_callback)
            # 情況 D：剛夾著番茄退回到預備點   
            elif self.current_step == 'LIFT':
                self.get_logger().info('[步驟 5] 退回點 A 完成！')
                self.current_step = 'RETURN'
                self.go_to_initial_joints(velocity=0.2, accel=0.2)
            # 情況 E：剛回到家 (Home)    
            elif self.current_step == 'RETURN':
                self.get_logger().info('完美完成任務！等待下一顆番茄。')
                self.control_gripper(0.0) 
                self.is_moving = False
                self.current_step = 'IDLE'
                # 🌟 加入這三行，廣播完成訊號告訴視覺大腦！
                
                self.status_pub.publish(msg)
        else:
            self.get_logger().warn(f'移動失敗，錯誤碼: {result.error_code.val}')
            self.is_moving = False
            self.current_step = 'IDLE'
            self.status_pub.publish(msg)
    # ==========================================================
    # 🌟 OMPL 大範圍移動輔助函數
    # ==========================================================
    def go_to_pose(self, x, y, z, qx, qy, qz, qw, yaw):
        self.is_moving = True 
        goal_msg = self.build_pose_goal_msg(x, y, z, qx, qy, qz, qw, yaw)
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def go_to_initial_joints(self, velocity=0.2, accel=0.2):
        self.is_moving = True
        #tset tomato ok?
        #target_degrees = [-90.0, 0.0, 0.0, 50.0, 90.0, 0.0]              #1
        #target_degrees = [-110.0, 0.0, 0.0, 50.0, 110.0, 0.0]            #2
        #target_degrees = [-70.0, 0.0, 0.0, 50.0, 70.0, 0.0]              #3
        #target_degrees = [-50.0, 0.0, 0.0, 50.0, 50.0, 0.0]              #4
        #target_degrees = [-90.0, -20.0, 150.0, -150.0, 90.0, 0.0]        #5
        #target_degrees = [-110.0, -20.0, 150.0, -150.0, 110.0, 0.0]      #6
        #target_degrees = [-90.0, -60.0, 150.0, -100.0, 90.0, 0.0]        #
        #target_degrees = [-50.0, -60.0, 150.0, -90.0, 50.0, 0.0]          #7
        #target_degrees = [-50.0, 70.0, 0.0, -70.0, -30.0, 0.0]           #8
        #target_degrees = [-130.0, 70.0, 0.0, -70.0, -150.0, 0.0]         #9
        #j1 = -90
        #target_degrees = [-90.0, 0.0, 0.0, 50.0, 90.0, 0.0]
        #target_degrees = [-90.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = -45
        #target_degrees = [-45.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [-45.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = 0
        #target_degrees = [0.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [0.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = 45
        #target_degrees = [45.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [45.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = 90
        #target_degrees = [90.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [90.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = 135
        #target_degrees = [135.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [135.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #j1 = 180
        #target_degrees = [180.0, 0.0, 0.0, 40.0, 90.0, 0.0]
        #target_degrees = [180.0, -20.0, 150.0, -150.0, 90.0, 0.0]
        #target_degrees = [0.0, 0.0, 0.0, 50.0, 90.0, 0.0]
        target_degrees = [0.0, -10.0, 40.0, 25.0, 90.0, 0.0]
        target_radians = [math.radians(deg) for deg in target_degrees]
        goal_msg = self.build_joint_goal_msg(target_radians, velocity, accel)
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.is_moving = False
            self.current_step = 'IDLE'
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def build_pose_goal_msg(self, x, y, z, qx, qy, qz, qw, yaw):
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name = 'tmr_arm'
        req.pipeline_id = 'ompl'
        req.planner_id = 'RRTstarkConfigDefault' 
        req.allowed_planning_time = 3.0  # 准許它算 3 秒鐘
        req.num_planning_attempts = 15
        req.max_velocity_scaling_factor = 0.15
        req.max_acceleration_scaling_factor = 0.15

        target_pose = Pose() # 設定目標位置
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        target_pose.orientation.x = float(qx)
        target_pose.orientation.y = float(qy)
        target_pose.orientation.z = float(qz)
        target_pose.orientation.w = float(qw)

        goal_constraints = Constraints() # 設定約束條件
        pos_con = PositionConstraint()   #位置約束：告訴手臂 flange (法蘭面) 必須抵達目標點
        pos_con.header.frame_id = 'base' 
        pos_con.link_name = 'flange' 
        s_primitive = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001]) # 誤差範圍 0.1 公分
        bv = BoundingVolume()
        bv.primitives.append(s_primitive)
        bv.primitive_poses.append(target_pose)
        pos_con.constraint_region = bv
        pos_con.weight = 1.0
        goal_constraints.position_constraints.append(pos_con)

        ori_con = OrientationConstraint() # 姿勢約束：告訴手臂夾爪的角度要對準
        ori_con.header.frame_id = 'base'
        ori_con.link_name = 'flange'
        ori_con.orientation = target_pose.orientation
        ori_con.absolute_x_axis_tolerance = 0.05 
        ori_con.absolute_y_axis_tolerance = 0.05
        ori_con.absolute_z_axis_tolerance = 0.05
        ori_con.weight = 1.0
        goal_constraints.orientation_constraints.append(ori_con)

        jc1 = JointConstraint()
        jc1.joint_name = 'joint_1'
        jc1.position = yaw  # 告訴它：J1 最完美的角度就是直接面對番茄
        jc1.tolerance_above = math.radians(60.0) # 給它左右各 60 度的彈性空間閃避奇異點
        jc1.tolerance_below = math.radians(60.0) # 但絕對不允許它轉超過 60 度 (徹底杜絕繞大圈)
        jc1.weight = 1.0
        goal_constraints.joint_constraints.append(jc1)
        
        jc3 = JointConstraint()
        jc3.joint_name = 'joint_3'
        jc3.position = math.radians(90.0)            # 基準點設在 90 度
        #jc3.tolerance_above = math.radians(89.0)    # 最高可以到 179 度
        #jc3.tolerance_below = math.radians(89.0)    # 最低只能到 1 度 (不准變負的)
        jc3.tolerance_above = math.radians(89.0)    
        jc3.tolerance_below = math.radians(89.0)    
        jc3.weight = 1.0
        goal_constraints.joint_constraints.append(jc3)
        '''
        jc4 = JointConstraint()
        jc4.joint_name = 'joint_4'
        jc4.position = math.radians(90.0)           # 基準點設在 90 度
        jc4.tolerance_above = math.radians(89.0)    # 最高可以到 179 度
        jc4.tolerance_below = math.radians(89.0)    # 最低只能到 1 度 (不准變負的)
        jc4.weight = 1.0
        goal_constraints.joint_constraints.append(jc4)
        
        jc5 = JointConstraint()
        jc5.joint_name = 'joint_5'
        jc5.position = math.radians(90.0)           
        jc5.tolerance_above = math.radians(20.0)    
        jc5.tolerance_below = math.radians(20.0)    
        jc5.weight = 1.0
        goal_constraints.joint_constraints.append(jc5)
        '''
        req.goal_constraints.append(goal_constraints)
        return goal_msg

    def build_joint_goal_msg(self, joint_angles, velocity, accel):
        goal_msg = MoveGroup.Goal()
        req = goal_msg.request
        req.group_name = 'tmr_arm'
        req.planner_id = 'RRTstarkConfigDefault'
        req.allowed_planning_time = 1.5
        req.max_velocity_scaling_factor, req.max_acceleration_scaling_factor = velocity, accel 

        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        goal_constraints = Constraints()
        for name, angle in zip(joint_names, joint_angles):
            jc = JointConstraint(joint_name=name, position=angle, tolerance_above=0.001, tolerance_below=0.001, weight=1.0)
            goal_constraints.joint_constraints.append(jc)
        req.goal_constraints.append(goal_constraints)
        return goal_msg

def main(args=None): 
    rclpy.init(args=args) # 初始化 ROS 2 系統
    node = TM5MControlNode() # 創建你的大腦節點
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