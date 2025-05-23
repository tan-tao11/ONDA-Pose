import json
import numpy
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

def save_json(cam_file, obj_file, rots, trans,scale, scene_id, obj_id):
    with open(cam_file, 'r') as f:
        poses_infos = json.load(f)
    target = dict()

    id = 0
    for key, value in poses_infos.items():
        frame_new = {"cam_R_m2c": list(np.reshape( R.from_quat(rots[id]).as_matrix(), 9)), "cam_t_m2c": list(trans[id]*1000), 'scale': scale, "obj_id": obj_id, 'scene_id': scene_id}
        target[key] = [frame_new]
        id += 1
    with open(obj_file, 'w') as f:
        json.dump(target, f, indent=4, ensure_ascii=False)

def save_json(opt_poses, bboxs_source, output_json, obj_id):
    with open(bboxs_source, 'r') as f:
        bbox_infos = json.load(f)

    for key, value in opt_poses.items():
        bbox_infos[f'{obj_id}/{key}'][0]['pose_refine'] = value.tolist()

    with open(output_json, 'w') as f:
        json.dump(bbox_infos, f, indent=4, ensure_ascii=False)

def extract_cam_poses(pose_file, train_list):
    with open(pose_file, 'r') as f:
        poses_infos = json.load(f)
    
    trans_cams = []
    quat_cams = []
    T_cams = []
    for key, value in poses_infos.items():
        if '{:06d}'.format(int(key)) not in train_list:
            continue
        rot = np.array(value[0]['cam_R_m2c'])
        rot = np.reshape(rot, [3, 3])
        trans = np.array(value[0]['cam_t_m2c'])/1000
        T = np.hstack([rot, trans[:, None]])
        # quat = R.from_matrix(rot).as_quat()
        # trans_cams.append(trans)
        # quat_cams.append(quat)
        T_cams.append(T)
    return np.array(T_cams)

def extract_obj_poses(file, split_list):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    quat_objs = []
    trans_objs = []
    for key, value in data.items():
        for key0, value0 in value.items():
            _, file_name = os.path.split(key0)
            if file_name.split('.')[0] not in split_list:
                continue
            Rot = value0['R']
            t = value0['t']
            quat = R.from_matrix(Rot).as_quat()
            quat_objs.append(quat)
            trans_objs.append(t[:, None])
    # print(np.array(quat_objs))
    return np.array(quat_objs), np.array(trans_objs)

def write_obj_poses(file, ref_file, rots, trans, scale):
    with open(ref_file, 'rb') as f:
        data_ref = pickle.load(f)
    index = 0 
    for key, value in data_ref.items():
        for key0, value0 in value.items():
            value0['R'] = R.from_quat(rots[index]).as_matrix()
            value0['t'] = trans[index]
            index += 1 
            data_ref[key][key0] = value0
    # print(data_ref)
    with open(file, 'wb') as f:
        data = pickle.dump(data_ref, f)