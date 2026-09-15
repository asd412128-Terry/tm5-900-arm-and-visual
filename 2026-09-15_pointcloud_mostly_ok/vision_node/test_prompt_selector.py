"""
獨立測試 TargetSelector.prompt_choose_id 的即時面板重繪，不需要 ROS/相機。
執行：python3 -m vision_node.test_prompt_selector
"""
import random

from .target_selector import TargetSelector


def make_target(x, y, z, occluded=False):
    return {'world_x': x, 'world_y': y, 'world_z': z, 'z_real': z,
            'vx': 0.0, 'vy': 0.0, 'vz': -1.0,
            'paired_tomato': {'occluded': occluded, 'occlusion_reason': '測試用遮擋', 'depth': z}}


def main():
    selector = TargetSelector()
    targets = [make_target(0.1, 0.2, 0.3), make_target(0.4, 0.1, 0.5, occluded=True)]
    valid, invalid_reasons, _ = selector.build_valid_candidates(targets)

    def refresh_fn(v):
        for t in targets:
            t['world_x'] += random.uniform(-0.005, 0.005)
            t['world_y'] += random.uniform(-0.005, 0.005)
        return selector.refresh_valid(v, targets, selector.max_reach_m)

    answer = selector.prompt_choose_id(valid, refresh_fn=refresh_fn, invalid_reasons=invalid_reasons)
    print(f"你選了: {answer!r}")


if __name__ == '__main__':
    main()
