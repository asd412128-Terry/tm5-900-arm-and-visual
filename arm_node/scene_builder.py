"""
============================================================================
 arm_node.scene_builder — 場景建構
============================================================================
 職責：把靜態障礙物、車體、虛擬夾爪寫進 MoveIt Planning Scene。

 座標約定：pos = Isaac Sim 的 Translate (x,y,z)，size = Isaac Sim 的 Scale (x,y,z)
============================================================================
"""
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from . import config


class SceneBuilder:
    def __init__(self, node: Node, scene_client):
        self.node = node
        self.scene_client = scene_client
        self.log = node.get_logger()

    def build_all(self):
        self.load_obstacles()
        if config.ENABLE_CAR_BODY:
            self.attach_car_body()
        if config.ENABLE_VIRTUAL_GRIPPER:
            self.attach_virtual_gripper()

    def load_obstacles(self):
        ok = 0
        for obs in config.OBSTACLES:
            co = CollisionObject()
            co.header.frame_id = config.BASE_FRAME
            co.id = obs['id']
            co.operation = CollisionObject.ADD

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in obs['pos']]
            pose.orientation.w = 1.0
            co.primitive_poses = [pose]

            prim = SolidPrimitive()
            if obs.get('type', 'box').lower() == 'cylinder':
                prim.type = SolidPrimitive.CYLINDER
            else:
                prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(d) for d in obs['size']]
            co.primitives = [prim]

            try:
                if self._apply_world(co):
                    self.log.info(f'✓ [{obs["id"]}] 已加入 Planning Scene')
                    ok += 1
            except Exception as e:
                self.log.error(f'載入障礙物 {obs["id"]} 失敗: {e}')

        self.log.info(f'障礙物載入完成：{ok}/{len(config.OBSTACLES)} 個成功')

    def attach_car_body(self):
        box = SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[float(d) for d in config.CAR_BODY_SIZE])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in config.CAR_BODY_OFFSET]
        pose.orientation.w = 1.0

        co = CollisionObject(id='car_body', operation=CollisionObject.ADD)
        co.header.frame_id = config.BASE_FRAME
        co.primitives, co.primitive_poses = [box], [pose]

        if self._apply_attached(co, link=config.BASE_FRAME, touch_links=config.CAR_TOUCH_LINKS):
            self.log.info(f'✓ 車體已掛載 {config.CAR_BODY_SIZE[0]} x {config.CAR_BODY_SIZE[1]} x {config.CAR_BODY_SIZE[2]} m')
        else:
            self.log.warn('車體掛載失敗')

    def attach_virtual_gripper(self):
        ok_all = True
        for name, link_name, sign in (('virtual_gripper_left', 'left_finger_link', -1.0),
                                       ('virtual_gripper_right', 'right_finger_link', +1.0)):
            prims, poses = [], []

            # 手指本體
            prims.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(config.VG_FINGER_SIZE)))
            fp = Pose()
            fp.position.x = sign * config.VG_FINGER_OFF_X
            fp.position.z = config.VG_FINGER_Z
            fp.orientation.w = 1.0
            poses.append(fp)

            # 手指延伸段
            prims.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(config.VG_FINGER_EXT_SIZE)))
            ep = Pose()
            ep.position.x = sign * config.VG_FINGER_OFF_X
            ep.position.z = config.VG_FINGER_EXT_Z
            ep.orientation.w = 1.0
            poses.append(ep)

            co = CollisionObject(id=name, operation=CollisionObject.ADD)
            co.header.frame_id = link_name
            co.primitives = prims
            co.primitive_poses = poses

            if not self._apply_attached(co, link=link_name, touch_links=config.VG_TOUCH_LINKS):
                ok_all = False

        if ok_all:
            self.log.info('✓ 虛擬夾爪已成功掛載（分開掛在左右手指，會跟著開合動）')
        else:
            self.log.warn('虛擬夾爪掛載失敗')

    def _apply_world(self, co: CollisionObject) -> bool:
        req = ApplyPlanningScene.Request()
        req.scene.world.collision_objects.append(co)
        req.scene.is_diff = True
        return self.scene_client.call(req).success

    def _apply_attached(self, co: CollisionObject, link: str, touch_links) -> bool:
        aco = AttachedCollisionObject(link_name=link, object=co)
        aco.touch_links = list(touch_links)
        req = ApplyPlanningScene.Request()
        req.scene.robot_state.attached_collision_objects.append(aco)
        req.scene.robot_state.is_diff = req.scene.is_diff = True
        return self.scene_client.call(req).success
