"""
============================================================================
 arm_node.math_utils — 向量幾何大腦
============================================================================
 職責：由「番茄座標 + 果梗 3D 方向向量」推出正交夾爪姿態、抓取點與預備點。

 輸入完全來自呼叫端（vision 給的番茄座標/果梗向量、arm 自己查的 TF 朝向），
 不碰任何 ROS topic / TF / MoveIt，是純函式、方便單獨測試。
============================================================================
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import config


class MathUtils:
    """向量幾何大腦：由果梗 3D 方向向量推出正交夾爪姿態、抓取點與預備點。"""

    @staticmethod
    def calculate_grasp_and_approach(tomato_x, tomato_y, tomato_z,
                                     stem_vec, base_yaw,
                                     R_current=None,
                                     gripper_length=config.GRIPPER_LENGTH,
                                     approach_dist=config.APPROACH_DIST):
        """
        利用 3D 果梗向量，建立正交夾爪座標系：
          z_axis: 接近軸，正交切入
          y_axis: 沿著果梗方向（對應夾爪實體的開合方向）
          x_axis: 正交於 Y 與 Z，維持右手定則

        ★ h 的來源：優先查 R_current(法蘭面現在實際朝向，從TF/URDF運動鏈FK查來)，
        取代原本純用 base_yaw 算出來的水平假設參考。R_current 查不到時(TF查詢
        失敗、或呼叫端沒傳)，自動退回原本 base_yaw 版本，不會崩潰。
        ★ 注意：這裡只是換 h 的來源，z_axis 依然是自由投影(沒有鉸鏈限制)，
        可行解範圍不受影響，只是換一個起點方向。
        """
        # ---- (a) 取得果梗方向向量 ----
        stem_v = np.array(stem_vec, dtype=float)
        norm_v = np.linalg.norm(stem_v)
        if norm_v < 1e-6:
            stem_dir = np.array([0.0, 0.0, -1.0])
        else:
            stem_dir = stem_v / norm_v

        # ---- (b) 定義參考向量 h：優先用法蘭面現在實際朝向，查不到才退回水平假設 ----
        if R_current is not None:
            h = np.asarray(R_current, dtype=float)[:, 2]
            hn = np.linalg.norm(h)
            h = h / hn if hn > 1e-9 else np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
        else:
            h = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])

        # ---- (c) 接近軸 (Z 軸)：將 h 投影到垂直果梗的平面 ----
        z_axis = h - np.dot(h, stem_dir) * stem_dir
        nz = np.linalg.norm(z_axis)
        if nz < 1e-6:
            # 退化情況：h 幾乎完全平行果梗方向 → 改拿世界 Z 軸當參考
            ref = np.array([0.0, 0.0, -1.0])
            z_axis = ref - np.dot(ref, stem_dir) * stem_dir
            z_axis /= np.linalg.norm(z_axis)
        else:
            z_axis = z_axis / nz

        # ---- (d) 讓 Y 軸對齊果梗反方向，透過外積求 X 軸維持右手定則 ----
        y_axis = -stem_dir
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # ---- (e) 組裝 3x3 旋轉矩陣並轉為 Quaternion ----
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()

        # ---- (f) 抓取點 / 預備點沿接近軸往回退 ----
        grasp = np.array([tomato_x, tomato_y, tomato_z]) - gripper_length * z_axis
        app   = grasp - approach_dist * z_axis

        grasp_target    = (grasp[0], grasp[1], grasp[2], qx, qy, qz, qw, base_yaw)
        approach_target = (app[0],   app[1],   app[2],   qx, qy, qz, qw, base_yaw)
        return grasp_target, approach_target

    """給某個座標+姿態，沿這個姿態的局部 Z 軸往回退 distance 公尺，回傳新的 (x,y,z)，
    姿態 (qx,qy,qz,qw) 不變。用在「已經知道要看向哪個點、哪個姿態，但要退到安全/適合
    拍攝的距離外」的情境——例如番茄座標配上一個朝向後，退到不會撞上去的距離。
    跟 calculate_grasp_and_approach 裡 `grasp - approach_dist * z_axis` 是同一套算法，
    這裡抽成獨立方法方便在番茄座標以外的情境重用（例如精定位目標）。"""
    @staticmethod
    def retreat_along_local_z(x, y, z, qx, qy, qz, qw, distance):
        z_axis = R.from_quat([qx, qy, qz, qw]).as_matrix()[:, 2]
        new_pos = np.array([x, y, z]) - distance * z_axis
        return (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

    """繞著目標點 T=(x,y,z)，把姿態 (qx,qy,qz,qw) 繞「世界 Z 軸」(垂直軸，且是繞
    T 這個點轉，不是繞原點)偏移 azimuth_offset_deg 度，再用轉過的新姿態對同一個 T
    呼叫 retreat_along_local_z 退 distance 公尺。因為只是把「姿態」整組繞世界垂直軸
    轉，新姿態的局部 Z 軸依然精確指向 T(面對目標無誤，局部 Z 軸的定義本來就是「鏡頭
    指向 T」)，退出來的新位置到 T 的距離也精確還是 distance(在以 T 為圓心、半徑
    distance 的球面上)；繞世界 Z 軸轉不影響任何向量的垂直分量，所以鏡頭的傾斜程度
    (站得直不直)也不變，只有水平方位角在轉，效果像鏡頭繞著 T 水平掃視。
    ★ joint_1 目標角 yaw：不是拿輸入的 yaw 直接加 azimuth_offset_deg。2026-09-04 實測
    發現這樣算跟實際 IK 解出來的 joint_1 差距很大（偏移角越大差越多，甚至方向都反了）
    ——因為「J1 轉多少度」只有繞著『基座自己的轉軸』轉時才等於方位角的偏移量，我們是
    繞著 T(不在基座正上方)轉，兩者不能直接畫等號。改用跟 arm_task_node.py
    _process_target 同一套公式 atan2(新位置.y, 新位置.x)——直接對「新算出來的鏡頭
    座標」算方位角，實測比對跟真實 IK 解出來的 joint_1 誤差只有幾度，且跟著偏移角
    增大也不會跑掉，比原本的加法準很多。這裡的 yaw 終究只是給 go_to_pose 的 J1 軟
    提示(J1_TOLERANCE 容差內即可)，不用是精確到小數點的真解。
    用在遮擋備用視角：跟主流程共用同一個目標點/退算函式，只換方位角。
    回傳 (x,y,z,qx,qy,qz,qw,yaw)。"""
    @staticmethod
    def orbit_around_target(x, y, z, qx, qy, qz, qw, distance, azimuth_offset_deg):
        theta = math.radians(azimuth_offset_deg)
        rot_offset = R.from_euler('z', theta)
        new_rot = rot_offset * R.from_quat([qx, qy, qz, qw])
        nqx, nqy, nqz, nqw = new_rot.as_quat()
        nx, ny, nz = MathUtils.retreat_along_local_z(x, y, z, nqx, nqy, nqz, nqw, distance)
        new_yaw = math.atan2(ny, nx)
        return (nx, ny, nz, float(nqx), float(nqy), float(nqz), float(nqw), new_yaw)
