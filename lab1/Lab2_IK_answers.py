import numpy as np
from scipy.spatial.transform import Rotation as R

def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def ccd_ik(meta_data, joint_positions, joint_orientations, target_pose):
    """CCD算法求解IK"""
    path, path_name, end_to_root, fixed_to_root = meta_data.get_path_from_root_to_end()
    
    # 以path为chain，start为根节点
    local_ori_list = []
    chain_offset_list = []
    for i, joint in enumerate(path):
        if i == 0:
            local_ori_list.append(R.from_quat(joint_orientations[joint]))
            chain_offset_list.append(np.zeros(3))
            continue
        parent_ori = local_ori_list[i - 1]
        local_ori_list.append(parent_ori.inv() * R.from_quat(joint_orientations[joint]))
        chain_offset_list.append(joint_positions[joint] - joint_positions[path[i - 1]])

    # 本地joint局部坐标系的ori
    max_iterate_times = 10
    min_deviation = 0.01
    link_start = path[0]   # 链式起点
    link_end = path[-1]    # 链式终点 也就是ik的端点
    end_pos = joint_positions[link_end]
    while max_iterate_times > 0:
        if distance(end_pos, target_pose) <= min_deviation:
            break
        # 从end进行一次遍历
        for i in range(len(path)-1, -1, -1):
            joint = path[i]
            if joint == link_end:
                continue
            parent = meta_data.joint_parent[joint]
            joint_pos = joint_positions[joint]
            radius = distance(joint_pos, end_pos)
            direction = (target_pose - joint_pos)
            # 计算出该joint旋转后，end_pos和target_pose距离最小的end_pos位置
            min_pos = direction * radius / np.linalg.norm(direction) + joint_pos
            # 计算该joint要执行的旋转
            old_vec = (end_pos - joint_pos) / radius
            new_vec = (min_pos - joint_pos) / radius
            rotation = R.from_rotvec(np.cross(old_vec, new_vec), degrees=False)
            local_ori_list[i] = rotation * local_ori_list[i]
            end_pos = min_pos

        # 执行一次chain的FK
        chain_ori_list = []
        chain_pos_list = []
        for j, joint in enumerate(path):
            if j == 0:
                chain_ori_list.append(local_ori_list[0])
                chain_pos_list.append(joint_positions[joint])
            else:
                chain_ori_list.append(chain_ori_list[j - 1] * local_ori_list[j])
                chain_pos_list.append(joint_positions[j - 1] + chain_ori_list[j - 1].apply(chain_offset_list[j]))
        # 对真正的root应用chain FK之后的结果
        root_idx = path.index(fixed_to_root[-1])
        joint_positions[0] = chain_pos_list[root_idx]
        joint_orientations[0] = chain_ori_list[root_idx].as_quat()
        # 执行一次FK
        for n in range(len(joint_orientations)):
            parent = meta_data.joint_parent[n]
            if parent == -1:
                parent = 0
            parent_ori = R.from_quat(joint_orientations[parent])
            if n in path and n != link_start:
                joint_orientations[n] = chain_ori_list[path.index(n)].as_quat()
            offset = meta_data.joint_initial_position[n] - meta_data.joint_initial_position[parent]
            joint_positions[n] = joint_positions[parent] + parent_ori.apply(offset)
            
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