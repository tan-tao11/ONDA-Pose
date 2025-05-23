import torch 
import numpy as np
import os.path as osp


def load_camera_pose(camera_path, idx):
    train_num = 0
    relative_rot_file = osp.join(camera_path, 'relative_rot.txt')
    relative_trans_file = osp.join(camera_path, 'relative_trans.txt')

    relative_rot = np.loadtxt(relative_rot_file).astype(np.float32)
    relative_trans = np.loadtxt(relative_trans_file).astype(np.float32)
    relative_rot = torch.from_numpy(relative_rot[idx])
    relative_trans = torch.from_numpy(relative_trans[idx])
    return relative_rot, relative_trans