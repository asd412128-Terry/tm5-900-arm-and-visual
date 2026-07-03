import rclpy
from rclpy.node import Node #（用來建立「大腦」節點
from rclpy.action import ActionClient #（用來跟手臂發送「動作任務」）
from rclpy.executors import MultiThreadedExecutor #（多執行緒引擎，讓大腦能同時做很多事）
from rclpy.callback_groups import ReentrantCallbackGroup #（允許多個任務同時執行的管理員）

from moveit_msgs.action import MoveGroup #（MoveIt 專用的通訊協定，告訴手臂你要移動到哪裡）
from moveit_msgs.msg import Constraints, OrientationConstraint, JointConstraint #（用來設定移動的限制條件）

class Tm5_900(Node):
    def __init__(self):
        super().__init__('Tm5_900_node') # 要是字串

        self.cb_group = ReentrantCallbackGroup() #（建立一個「多線道」，讓多個任務可以同時執行）

        #（建立一個「動作客戶端」，專門用來跟手臂說「我要移動了」）
        self.move_client = ActionClient(self, MoveGroup, 'move_action',callback_group = self.cb_group)

        self.get_logger().info('正在連接 MoveIt 動作伺服器...')
        self.timer = self.create_timer(1.0, self.check_connection) #（每秒檢查一次連線狀態）
        
    def check_connection(self):
        if self.move_client.server_is_ready():
            self.get_logger().info('成功連接到 MoveIt 動作伺服器！')
            self.timer.cancel()
            self.send_goal_joints()
        else:
            self.get_logger().error('無法連接到 MoveIt 動作伺服器！請確保 MoveIt 已經啟動。')

    def send_goal_joints(self):
        goal_msg = MoveGroup.Goal() #（建立一個「動作任務」的空殼）
        goal_msg.request.group_name = 'tmr_arm' # 手臂群組名稱
        
        target_radians = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0] #（目標關節角度，單位是弧度）
        goal_constraints = Constraints() #（建立一個總約束容器）

        for i ,angle in enumerate(target_radians):
            jc = JointConstraint() #（建立一個關節約束）
            jc.joint_name = f'joint_{i+1}'
            jc.position = float(angle)
            jc.tolerance_above = jc.tolerance_below = 0.01 #（允許的誤差範圍，單位是弧度）
            jc.weight = 1.0 #（這個約束的重要程度，1.0 是最重要）
            goal_constraints.joint_constraints.append(jc) #（把這個關節約束加到總約束容器裡）

        goal_msg.request.goal_constraints.append(goal_constraints)
        self.get_logger().info('正在發送六個關節角度...')
        send_goal_future = self.move_client.send_goal_async(goal_msg) # 發出指令，不等待，拿回一個憑據
        send_goal_future.add_done_callback(self.goal_response_callback) # 接到回應後要做什麼

    def goal_response_callback(self, future): # 確認 Server 有沒有接單
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt 動作任務被拒絕了！')
            return
        
        self.get_logger().info('MoveIt 動作任務已接受，正在執行...')
        get_result_future = goal_handle.get_result_async() # 等待結果回傳
        get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future): # 等到任務完成後的回應
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info('MoveIt 動作任務成功完成！')
        else:
            self.get_logger().error(f'MoveIt 動作任務失敗，錯誤代碼: {result.error_code.val}')

def main(args = None):
    rclpy.init(args = args)
    Tm5_900node = Tm5_900() # 利用 Tm5_900 類別，「建立」出一個物件實體，存進 Tm5_900node
    executor = MultiThreadedExecutor() # 建立執行器（引擎）
    executor.add_node(Tm5_900node) # 把大腦放進引擎

    try:
        executor.spin() # 開始運轉
    except KeyboardInterrupt: # crtl c
        pass
    finally:
        Tm5_900node.destroy_node() # 結束前把大腦關掉
        rclpy.shutdown() # 結束前把 ROS 也關掉

if __name__ == '__main__':
    main()