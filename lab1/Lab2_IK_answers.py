import numpy as np
from scipy.spatial.transform import Rotation as R

def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def ccd_ik(meta_data, joint_positions, joint_orientations, target_pose):
    """CCD算法求解IK"""
    path, path_name, end_to_root, fixed_to_root = meta_data.get_path_from_root_to_end()
    
    joint_init_local_ori = [R.from_euler('xyz', [0, 0, 0], degrees=True)] + [R.from_quat(joint_orientations[meta_data.joint_parent[joint]]).inv() * R.from_quat(ori) for joint,ori in enumerate(joint_orientations) if joint > 0]
    # 以path为chain，start为根节点
    local_ori_list = []
    update_loacal_ori_list = [R.from_euler('xyz', [0, 0, 0], degrees=True)] * len(path)
    chain_offset_list = []
    for i, joint in enumerate(path):
        if i == 0:
            local_ori_list.append(R.from_quat(joint_orientations[joint]))
            chain_offset_list.append(np.zeros(3))
            continue
        parent_ori = R.from_quat(joint_orientations[path[i-1]])
        local_ori_list.append(parent_ori.inv() * R.from_quat(joint_orientations[joint]))
        chain_offset_list.append(joint_positions[joint] - joint_positions[path[i - 1]])

    # 本地joint局部坐标系的ori
    max_iterate_times = 30
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
            if joint == link_end or joint == 0:
                continue
            joint_pos = joint_positions[joint]
            radius = distance(joint_pos, end_pos)
            direction = (target_pose - joint_pos)
            # 计算出 该joint旋转后，end_pos和target_pose距离最小的end_pos位置
            min_pos = (direction * radius / np.linalg.norm(direction)) + joint_pos
            # 计算该joint要执行的旋转
            old_vec = (end_pos - joint_pos) / radius
            new_vec = (min_pos - joint_pos) / radius
            rotation = R.from_rotvec(np.cross(old_vec, new_vec), degrees=False)
            update_loacal_ori_list[i] = rotation * update_loacal_ori_list[i]
            end_pos = min_pos

        # 执行一次chain的FK
        chain_ori_list = []
        chain_pos_list = []
        for j, joint in enumerate(path):
                
            if j == 0:
                chain_ori_list.append(local_ori_list[0])
                chain_pos_list.append(joint_positions[joint])
                continue
            chain_ori_list.append(chain_ori_list[j - 1] * update_loacal_ori_list[j] * local_ori_list[j])
            chain_pos_list.append(chain_pos_list[j - 1] + chain_ori_list[j - 1].apply(chain_offset_list[j]))
        for i in range(len(path)):
            joint_positions[path[i]] = chain_pos_list[i]
            joint_orientations[path[i]] = chain_ori_list[i].as_quat()
        # # 执行一次FK
        for n in range(1, len(joint_orientations)):
            parent = meta_data.joint_parent[n]
            parent_ori = R.from_quat(joint_orientations[parent])
            if n in path:
             if n in fixed_to_root:
                joint_orientations[n] = (parent_ori * update_loacal_ori_list[path.index(n)-1].inv() * joint_init_local_ori[n]).as_quat()
            else:
                joint_orientations[n] = (parent_ori * joint_init_local_ori[n]).as_quat()
            offset = meta_data.joint_initial_position[n] - meta_data.joint_initial_position[parent]
            joint_positions[n] = joint_positions[parent] + parent_ori.apply(offset)
            
        max_iterate_times -= 1
    return joint_positions, joint_orientations



def ppp_ik(meta_data, joint_positions, joint_orientations, target_pose):
    def get_joint_rotations():
        # 相对父节点的ori
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        # 相对父节点的pos
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1 and path2[0] != 0:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path),), dtype=object)       # 局部旋转
    position_chain = np.empty((len(path), 3))                   # 全局坐标
    orientation_chain = np.empty((len(path),), dtype=object)    # 全局旋转
    offset_chain = np.empty((len(path), 3))                     # offset

    # 对chain进行初始化
    if len(path2) > 1:
        orientation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        orientation_chain[0] = R.from_quat(joint_orientations[path[0]])

    position_chain[0] = joint_positions[path[0]]
    rotation_chain[0] = orientation_chain[0]
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        position_chain[i] = joint_positions[index]
        if index in path2:
            # essential
            orientation_chain[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_chain[i] = R.from_quat(joint_rotations[index]).inv()
            offset_chain[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            orientation_chain[i] = R.from_quat(joint_orientations[index])
            rotation_chain[i] = R.from_quat(joint_rotations[index])
            offset_chain[i] = joint_offsets[index]


    # CCD IK
    times = 10
    distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
    while times > 0 and distance > 0.001:
        times -= 1
        # 先动手
        for i in range(len(path) - 2, -1, -1):
        # 先动腰
        # for i in range(1, len(path) - 1):
            if joint_parent[path[i]] == -1:
                continue
            cur_pos = position_chain[i]
            # 计算旋转的轴角表示
            c2t = target_pose - cur_pos
            c2e = position_chain[-1] - cur_pos
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            # 由于float的精度问题，cos可能cos(theta)可能大于1.
            cos = min(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)), 1.0)
            theta = np.arccos(cos)
            # 防止quat为0？
            if theta < 0.0001:
                continue
            delta_rotation = R.from_rotvec(theta * axis)
            # 更新当前的local rotation 和子关节的position, orientation
            orientation_chain[i] = delta_rotation * orientation_chain[i]
            rotation_chain[i] = orientation_chain[i - 1].inv() * orientation_chain[i]
            for j in range(i + 1, len(path)):
                orientation_chain[j] = orientation_chain[j - 1] * rotation_chain[j]
                position_chain[j] = np.dot(orientation_chain[j - 1].as_matrix(), offset_chain[j]) + position_chain[j - 1]
            distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))


    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_chain[i]
        if index in path2:
            joint_rotations[index] = rotation_chain[i].inv().as_quat()
        else:
            joint_rotations[index] = rotation_chain[i].as_quat()

    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_chain[0]).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_chain[root_index].as_quat()
            joint_positions[0] = position_chain[root_index]


    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])


    return joint_positions, joint_orientations


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动1学
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