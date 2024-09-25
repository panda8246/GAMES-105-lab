import numpy as np
from scipy.spatial.transform import Rotation as R

def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def ccd_ik(meta_data, joint_positions, joint_orientations, target_pose):
    """CCD算法求解IK"""
    path, path_name, end_to_root, fixed_to_root = meta_data.get_path_from_root_to_end()
    max_iterate_times = 5
    min_deviation = 0.01
    link_start = path[0]   # 链式起点
    link_end = path[-1]    # 链式终点 也就是ik的端点
    while max_iterate_times > 0:
        end_pos = joint_positions[link_end]
        if distance(end_pos, target_pose) <= min_deviation:
            break
        # 从end进行一次遍历
        for i in range(len(path)-1, 0, -1):
            joint = path[i]
            # 这里的parent并不是骨骼结构上的父关节，而是链式结构上的前一个关节
            parent_joint = path[i-1]
            joint_pos = joint_positions[joint]
            joint_ori = joint_orientations[joint]
            parent_joint_pos = joint_positions[parent_joint]
            length = distance(parent_joint_pos, joint_pos)
            direction = (target_pose - joint_pos)
            min_pos = direction * length / np.linalg.norm(direction)

        max_iterate_times -= 1
    return joint_positions, joint_orientations


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    
    return ccd_ik(meta_data, joint_positions, joint_orientations, target_pose)
    

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations