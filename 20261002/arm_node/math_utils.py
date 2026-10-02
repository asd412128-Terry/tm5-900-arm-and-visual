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

        # ★ 2026-09-23：J1 約束目標角改用 atan2(預備點A.y, 預備點A.x)，不再直接用
        # base_yaw（番茄座標自己的方位角）。跟 orbit_around_target/
        # camera_facing_flange_pose 的 yaw 算法同一套慣例：對「最終算出來的法蘭
        # 座標」算方位角，不是對「目標物座標」算方位角。
        # 原因：備用視角是把接近方向繞目標轉 ALT_VIEW_AZIMUTH_OFFSETS_DEG(±50°)，
        # 這個轉動量已經大於 J1_TOLERANCE(±40°)——只要「實際需要的 J1 偏移」大致跟著
        # 這個轉動量走（合理，因為整個接近方向都繞著目標轉了），拿沒轉動前的
        # base_yaw ±40° 當窗口，幾乎必然把真實 IK 解排除在外，不是某一顆番茄運氣不好，
        # 是走備用視角時的系統性問題。實測案例：base_yaw 算出的窗口漏掉真實 IK 解的
        # J1（差約 8.5°，跟「轉動量 50° 超過容忍度 40°」量級吻合），改成量測「預備點A
        # 自己的方位角」（已經吃進 R_current/z_axis 帶的旋轉）之後，窗口正確涵蓋。
        yaw_hint = math.atan2(app[1], app[0])
        grasp_target    = (grasp[0], grasp[1], grasp[2], qx, qy, qz, qw, yaw_hint)
        approach_target = (app[0],   app[1],   app[2],   qx, qy, qz, qw, yaw_hint)
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

    """接近方向 a → 法蘭姿態旋轉矩陣：局部 Z 軸 = a，局部 Y 軸盡量朝世界向上，局部 X = Y × Z。
    a 垂直朝上/下時無法定義「向上」，丟 ValueError。"""
    @staticmethod
    def look_at_rotation(a):
        z_axis = np.asarray(a, dtype=float)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
        n = np.linalg.norm(x_axis)
        if n < 1e-6:
            raise ValueError('接近方向垂直朝上/下，無法決定姿態的「向上」方向')
        x_axis = x_axis / n
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    """粗定位（車載相機）給番茄中心 T=(tx,ty,tz) 與水平接近方向 a=(ax,ay,az)，算出「相機」
    面對 T、光軸通過 T、離 T distance 公尺時，法蘭(link_6，末端中心)該去的
    (x,y,z,qx,qy,qz,qw,yaw)。
    跟 retreat_along_local_z 的差別：那個以「法蘭 Z 軸」為準，但手臂相機不在法蘭軸線上
    （側向偏移約 13cm，光軸還跟法蘭 Z 差約 5.6°），法蘭 Z 對準番茄的話相機會偏離番茄。
    這裡以「相機光軸」為準：
      1. 先用 look_at_rotation(a) 擺好法蘭姿態 r0（Y 向上，跟既有姿態慣例一致，不影響畫面正反）
      2. 相機光軸在世界座標的方向 = r0 @ cam_z，再用最小旋轉把它轉到剛好等於 a（吃掉傾斜）
      3. 相機位置 = T − distance × a；法蘭位置 = 相機位置 − r_fl @ cam_t
    cam_t / cam_z：link_6 座標系下的相機位置 (3,) 與光軸方向 (3,)，呼叫端查 TF
    (ARM_FLANGE_FRAME → CAMERA_OPTICAL_FRAME) 取得。光軸(Z 軸)不受 isaac 反投影 x/y 手動
    翻轉影響，所以 real/isaac 兩種模式的 TF 都能直接用。
    azimuth_offset_deg：把接近方向 a 繞「經過 T 的世界 Z 軸」轉這個角度，用在遮擋備用視角
    （效果跟 orbit_around_target 一樣：距離跟面對目標保證不變，只換方位角）。
    yaw 沿用 orbit_around_target 的算法：atan2(法蘭位置.y, 法蘭位置.x)，只是 J1 軟提示。"""
    @staticmethod
    def camera_facing_flange_pose(tx, ty, tz, ax, ay, az, distance, cam_t, cam_z,
                                   azimuth_offset_deg=0.0):
        a = np.array([ax, ay, az], dtype=float)
        a = R.from_euler('z', math.radians(azimuth_offset_deg)).apply(a / np.linalg.norm(a))
        r0 = MathUtils.look_at_rotation(a)

        cam_z = np.asarray(cam_z, dtype=float)
        zc = r0 @ (cam_z / np.linalg.norm(cam_z))
        axis = np.cross(zc, a)
        s, c = np.linalg.norm(axis), float(np.dot(zc, a))
        if s > 1e-9:
            r_align = R.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()
            r_fl = r_align @ r0
        else:
            r_fl = r0

        p_cam = np.array([tx, ty, tz]) - distance * a
        p_fl = p_cam - r_fl @ np.asarray(cam_t, dtype=float)
        qx, qy, qz, qw = R.from_matrix(r_fl).as_quat()
        yaw = math.atan2(p_fl[1], p_fl[0])
        return (float(p_fl[0]), float(p_fl[1]), float(p_fl[2]),
                float(qx), float(qy), float(qz), float(qw), yaw)
