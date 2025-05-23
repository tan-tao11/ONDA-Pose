import argparse
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from threading import Thread
from omegaconf import OmegaConf
from pathlib import Path

from .utils.common import load_points
from .utils.pose_io import extract_cam_poses, extract_obj_poses, write_obj_poses, save_json


class Estimator_Step():
    def __init__(self, cfg):
        self.cfg = cfg
        self.obj_file = cfg.obj_pose_file
        self.cam_file = cfg.cam_pose_file
        self.ply_file = cfg.model_ply_file
        self.scene_split = cfg.scene_split
        self.scene_id = cfg.scene_id
        self.obj_id = cfg.obj_id
        self.init_scale = cfg.scale
        self.num_threads = cfg.threads
        self.use_symmetric = cfg.symmetric

        self.points_3d = load_points(self.ply_file, 0.4)
        self.output_json = self.obj_file.replace(".pkl", ".json")

    class ReprojectionThread(Thread):
        def __init__(self, xyz_i, kdtree):
            super().__init__()
            self.xyz_i = xyz_i
            self.kdtree = kdtree
            self.result = None

        def run(self):
            distances, _ = self.kdtree.query(self.xyz_i, k=2)
            self.result = np.mean(distances[:, 1])

        def get_result(self):
            return self.result

    def reprojection_error(self, x):
        total_error = 0.0
        rots = R.from_quat(x[:self.num_cam * 4].reshape(-1, 4)).as_matrix()
        transes = x[self.num_cam * 4:-1].reshape(-1, 3, 1)
        scale = x[-1]

        points_h = np.concatenate([self.points_3d, np.ones((self.points_3d.shape[0], 1))], axis=1)
        coords_xyz = []

        for i in range(self.num_cam):
            obj_pose = np.hstack((rots[i], transes[i]))
            obj_pose = np.vstack((obj_pose, [0, 0, 0, 1]))
            cam_pose = np.vstack((self.camera_poses[i], [0, 0, 0, 1]))
            cam_pose[:3, 3] *= scale
            proj_mat = np.linalg.inv(cam_pose) @ obj_pose
            projected = (proj_mat @ points_h.T).T
            projected /= projected[:, 3][:, None]
            coords_xyz.append(projected)

        coords_xyz = np.array(coords_xyz)
        mean_xyz = np.mean(coords_xyz, axis=0)
        for i in range(self.num_cam):
            error = np.linalg.norm(mean_xyz - coords_xyz[i], ord=1, axis=1)
            total_error += np.sum(error) / self.points_3d.shape[0]

        # print("[Non-symmetric] Total error:", total_error / self.num_cam)
        return total_error / self.num_cam

    def reprojection_error_symmetric(self, x):
        total_error = 0.0
        rots = R.from_quat(x[:self.num_cam * 4].reshape(-1, 4)).as_matrix()
        transes = x[self.num_cam * 4:-1].reshape(-1, 3, 1)
        scale = x[-1]

        points_h = np.concatenate([self.points_3d, np.ones((self.points_3d.shape[0], 1))], axis=1)
        coords_xyz = []

        for i in range(self.num_cam):
            obj_pose = np.hstack((rots[i], transes[i]))
            obj_pose = np.vstack((obj_pose, [0, 0, 0, 1]))
            cam_pose = np.vstack((self.camera_poses[i], [0, 0, 0, 1]))
            cam_pose[:3, 3] *= scale
            proj_mat = np.linalg.inv(cam_pose) @ obj_pose
            projected = (proj_mat @ points_h.T).T
            projected /= projected[:, 3][:, None]
            coords_xyz.append(projected[:, :3])

        coords_xyz = np.array(coords_xyz)

        for i in range(0, self.num_cam, self.num_threads):
            pool = []
            for j in range(self.num_threads):
                idx = i + j
                if idx >= self.num_cam:
                    break
                xyz_others = np.delete(coords_xyz, idx, axis=0).reshape(-1, 3)
                kdtree = KDTree(xyz_others)
                t = self.ReprojectionThread(coords_xyz[idx], kdtree)
                t.start()
                pool.append(t)

            for t in pool:
                t.join()
                total_error += t.get_result()

        # print("[Symmetric] Total error:", total_error / self.num_cam)
        return total_error / self.num_cam

    def run(self):
        scene_split_path = Path(self.scene_split)
        split_root = scene_split_path.parent
        split_base_name = scene_split_path.name

        splits = os.listdir(split_root)
        splits = [s for s in splits if split_base_name in s]
        splits.sort()

        opt_poses = {}
        for split in splits:
            split_file = os.path.join(split_root, split, "train.txt")
            with open(split_file, 'r') as f:
                train_list = f.readlines()
                train_list = [line.rstrip() for line in train_list]
                train_list = [line.split(' ')[-1] for line in train_list]

            self.initial_rots, self.initial_trans = extract_obj_poses(self.obj_file, train_list)
            self.camera_poses = extract_cam_poses(self.cam_file, train_list)
            self.num_cam = len(self.camera_poses)
            self.init_params = np.concatenate([self.initial_rots.reshape(-1), self.initial_trans.reshape(-1), [self.init_scale]])

            opt_rots, opt_trans = self.run_split()

            for idx, key in enumerate(train_list):
                opt_poses[str(int(key))] = np.hstack(( R.from_quat(opt_rots[idx]).as_matrix(), opt_trans[idx].reshape([3,1])/1000)) 

        # Save optimized poses
        save_json(opt_poses, self.cfg.bboxs_source, self.cfg.output_json, self.cfg.obj_id)

    def run_split(self):
        start_time = time.time()
        error_fn = self.reprojection_error_symmetric if self.use_symmetric else self.reprojection_error

        result = minimize(error_fn, self.init_params, method="L-BFGS-B", tol=1e-10,
                          options={"maxiter": 50, "maxfun": 1500000, "disp": True})
        # result = minimize(error_fn, self.init_params, method="L-BFGS-B", tol=1e-10,
        #                   options={"maxiter": 30, "maxfun": 50000, "disp": True})

        opt_pose = result.x
        opt_rots = opt_pose[:self.num_cam * 4].reshape(-1, 4)
        opt_trans = opt_pose[self.num_cam * 4:-1].reshape(-1, 3)
        opt_scale = opt_pose[-1]

        print("Optimized rotations:\n", opt_rots)
        print("Optimized translations:\n", opt_trans)
        print("Optimized scale:", opt_scale)
        print("Total runtime: {:.2f}s".format(time.time() - start_time))

        # write_obj_poses('lm_lm_real_ape_pred_init_preds.pkl', self.obj_file, opt_rots, opt_trans, opt_scale)
        return opt_rots, opt_trans

        
