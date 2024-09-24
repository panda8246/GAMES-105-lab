import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def load_joint_data(bvh_file_path):
    """
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        stack = []
        # 忽略第一行的 HIERACRCHY
        for i in range(1, len(lines)):
            # 读取完毕，退出
            line = lines[i].strip()
            if line.startswith('MOTION'):
                break
            if line.startswith('ROOT') or line.startswith('JOINT'):
                joint_name.append(line[5:].strip())
                if not stack:
                    joint_parent.append(-1)
                else:
                    joint_parent.append(stack[-1])
            elif line.startswith('End Site'):
                parent_name = joint_name[stack[-1]]
                joint_name.append(parent_name + "_end")
                joint_parent.append(stack[-1])
            elif line.startswith('OFFSET'):
                pos = [float(x) for x in line[6:].split()]
                joint_offset.append(pos)
            elif line.startswith('{'):
                stack.append(len(joint_name) - 1)
            elif line.startswith('}'):
                stack.pop()
    return joint_name, joint_parent, joint_offset  


def part1_calculate_T_pose(bvh_file_path):
    return load_joint_data(bvh_file_path)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    frame_data = motion_data[frame_id]
    channels_data = []
    # 将数组成3个float一组的自由度
    for i in range(0, len(frame_data), 3):
        channels_data.append(np.array(frame_data[i:i+3]))
    root_pos = channels_data[0]
    root_ori = channels_data[1]
    channels_data = channels_data[1:]
    joint_positions.append(root_pos)
    joint_orientations.append(R.from_euler('XYZ', root_ori, degrees=True).as_quat())
    # 计算FK过程
    ori_idx = 0
    for i, parent in enumerate(joint_parent):
        if parent == -1:
            continue
        # 末端节点没有自由度
        if not joint_name[i].endswith("_end"):
            ori_idx += 1
        parent_pos = joint_positions[parent]
        parent_ori = joint_orientations[parent]
        parent_ori = R.from_quat(parent_ori)
        # P = Pp + Qp * Oi
        cur_pos = parent_pos + parent_ori.apply(joint_offset[i])
        joint_positions.append(cur_pos)
        if not joint_name[i].endswith("_end"):
            # Q = Qp * Qi
            cur_ori = (parent_ori * R.from_euler('XYZ', channels_data[ori_idx], degrees=True)).as_quat()
        else:
            # 末端节点沿用父节点的朝向
            cur_ori = parent_ori.as_quat()
        joint_orientations.append(cur_ori)
    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_names, T_parents, T_offsets = load_joint_data(T_pose_bvh_path)
    A_names, A_parents, A_offsets = load_joint_data(A_pose_bvh_path)
    # 过滤_end
    T_names = [name for name in T_names if not name.endswith("_end")]
    A_names = [name for name in A_names if not name.endswith("_end")]
    A_motion = load_motion_data(A_pose_bvh_path)
    # 获取起始帧的TPose和APose
    A_start_frame = A_motion[0]
    new_motion = A_motion.copy()
    for a_i, a_name in enumerate(A_names):
        if a_name not in ["lShoulder", "rShoulder"]:
            continue
        t_i = T_names.index(a_name)
        a_ori = R.from_euler('XYZ', A_start_frame[(a_i + 1)*3:(a_i + 2)*3], degrees=True)
        # 求得 A->T 的变换
        a2t_trans = a_ori.inv()
        t2a_trans = a2t_trans.inv()
        for frame_id, frame_data in enumerate(A_motion):
            start_iex = (a_i + 1) * 3
            old_data = R.from_euler('XYZ', frame_data[start_iex:start_iex+3], degrees=True)
            new_motion[frame_id][start_iex:start_iex+3] = (a2t_trans * old_data).as_euler('XYZ', degrees=True)
    return new_motion

def part3_retarget_func22(T_pose_bvh_path, A_pose_bvh_path):
    def index_bone_to_channel(index, flag):
        if flag == 't':
            end_bone_index = end_bone_index_t
        else:
            end_bone_index = end_bone_index_a
        for i in range(len(end_bone_index)):
            if end_bone_index[i] > index:
                return index - i
        return index - len(end_bone_index)
    
    def get_t2a_offset(bone_name):
        l_bone = ['lShoulder', 'lElbow', 'lWrist'] 
        r_bone = ['rShoulder', 'rElbow', 'rWrist']
        if bone_name in l_bone:
            return R.from_euler('XYZ', [0.,0.,45.], degrees=True)
        if bone_name in r_bone:
            return R.from_euler('XYZ', [0.,0.,-45.], degrees=True)
        return R.from_euler('XYZ', [0.,0.,0.], degrees=True)


    motion_data = load_motion_data(A_pose_bvh_path)

    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)

    end_bone_index_t = []
    for i in range(len(t_name)):
        if t_name[i].endswith('_end'):
            end_bone_index_t.append(i)

    end_bone_index_a = []
    for i in range(len(a_name)):
        if a_name[i].endswith('_end'):
            end_bone_index_a.append(i)

    for m_i in range(len(motion_data)):
        frame = motion_data[m_i]
        cur_frame = np.empty(frame.shape[0])
        cur_frame[:3] = frame[:3]
        for t_i in range(len(t_name)):
            cur_bone = t_name[t_i]
            a_i = a_name.index(t_name[t_i])
            if cur_bone.endswith('_end'):
                continue
            channel_t_i = index_bone_to_channel(t_i, 't')
            channel_a_i = index_bone_to_channel(a_i, 'a')
            
            # retarget
            local_rotation = frame[3+channel_a_i*3 : 6+channel_a_i*3]
            if cur_bone in ['lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow', 'rWrist']:
                p_bone_name = t_name[t_parent[t_i]]
                Q_pi = get_t2a_offset(p_bone_name)
                Q_i = get_t2a_offset(cur_bone)
                local_rotation = (Q_pi * R.from_euler('XYZ', local_rotation, degrees=True) * Q_i.inv()).as_euler('XYZ', degrees=True)
            cur_frame[3+channel_t_i*3 : 6+channel_t_i*3] = local_rotation

        motion_data[m_i] = cur_frame

    return motion_data

# test

import os.path as osp
# joint_name, joint_parent, joint_offset = part1_calculate_T_pose(osp.abspath(osp.dirname(__file__)) + "/data/walk60.bvh")
# motion_data = load_motion_data(osp.abspath(osp.dirname(__file__)) + "/data/walk60.bvh")
# joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
part3_retarget_func(osp.abspath(osp.dirname(__file__)) + "/data/walk60.bvh", osp.abspath(osp.dirname(__file__)) + "/data/A_pose_run.bvh")