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
