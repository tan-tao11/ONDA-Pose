# modified from tensorpack/utils/utils.py
import os
import os.path as osp
import sys
import torch
import math
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.text import MIMEText
import inspect
from inspect import getframeinfo, stack
import numpy as np
import smtplib
import shutil
import pickle
import string
from termcolor import colored
from tqdm import tqdm
from . import logger
import copy
import functools
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat

cur_dir = osp.normpath(osp.abspath(osp.dirname(__file__)))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../"))
# __all__ = [
#     "change_env",
#     "get_rng",
#     "fix_rng_seed",
#     "get_tqdm",
#     "execute_only_once",
#     "humanize_time_delta",
#     "get_time_str",
#     "backup_path",
#     "argsort_for_list",
# ]


def msg(*args, sep=" "):
    # like print, but return a string
    return sep.join("{}".format(a) for a in args)


def lazy_property(function):
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def iiprint(*args, **kwargs):
    # print_info
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.relpath(caller.filename, PROJ_ROOT)
        if len(caller.filename) < len(filename):
            filename = caller.filename
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "cyan")
        print(date + " " + " ".join(map(str, args)), **kwargs)


def iprint(*args, **kwargs):
    # print_info, only basename
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.basename(caller.filename).split(".")[0]
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "cyan")
        print(date + " " + " ".join(map(str, args)), **kwargs)


def ddprint(*args, **kwargs):
    # print for debug
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.relpath(caller.filename, PROJ_ROOT)
        if len(caller.filename) < len(filename):
            filename = caller.filename
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "yellow")
        print(date + " " + colored("DBG ", "yellow", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def dprint(*args, **kwargs):
    # print for debug, only basename
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.basename(caller.filename).split(".")[0]
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "yellow")
        print(date + " " + colored("DBG ", "yellow", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def wwprint(*args, **kwargs):
    # print for warn
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.relpath(caller.filename, PROJ_ROOT)
        if len(caller.filename) < len(filename):
            filename = caller.filename
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "magenta")
        print(date + " " + colored("WRN ", "magenta", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def wprint(*args, **kwargs):
    # print for warn, only basename
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.basename(caller.filename).split(".")[0]
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "magenta")
        print(date + " " + colored("WRN ", "magenta", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def eeprint(*args, **kwargs):
    # print for error
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.relpath(caller.filename, PROJ_ROOT)
        if len(caller.filename) < len(filename):
            filename = caller.filename
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "red")
        print(date + " " + colored("ERR ", "red", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def eprint(*args, **kwargs):
    # print for error, only basename
    if True:
        caller = getframeinfo(stack()[1][0])
        filename = osp.basename(caller.filename).split(".")[0]
        date = colored("[{}@{}:{}]".format(get_time_str("%m%d_%H%M%S"), filename, caller.lineno), "red")
        print(date + " " + colored("ERR ", "red", attrs=["blink"]) + " ".join(map(str, args)), **kwargs)


def update_cfg(base_cfg, update_cfg):
    """used for mmcv.Config or other dict-like configs."""
    res_cfg = copy.deepcopy(base_cfg)
    res_cfg.update(update_cfg)
    return res_cfg


def f(f_string):
    """mimic fstring (in python >= 3.6) for python < 3.6."""
    frame = inspect.stack()[1][0]
    return Formatter(frame.f_globals, frame.f_locals).format(f_string)


class Formatter(string.Formatter):
    def __init__(self, globals_, locals_):
        self.globals = globals_
        self.locals = locals_

    def _vformat(self, *args, **kwargs):
        _vformat = super(Formatter, self)._vformat
        if "auto_arg_index" in inspect.getargspec(_vformat)[0]:
            kwargs["auto_arg_index"] = False
        return _vformat(*args, **kwargs)

    def get_field(self, field_name, args, kwargs):
        if not field_name.strip():
            raise ValueError("empty expression not allowed")
        return eval("(" + field_name + ")", self.globals, self.locals), None


def argsort_for_list(s, reverse=False):
    """get index for a sorted list."""
    return sorted(range(len(s)), key=lambda k: s[k], reverse=reverse)


def backup_path(path, backup_name=None):
    """backup a path if exists."""
    if os.path.exists(path):
        if backup_name is None or os.path.exists(backup_name):
            backup_name = path + "." + get_time_str()
        shutil.move(path, backup_name)
        logger.info("Existing path '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821, E501


def get_time_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)


# def get_time_str(fmt='%Y%m%d_%H%M%S'):
#     # from mmcv.runner import get_time_str
#     return time.strftime(fmt, time.localtime())  # defined in mmcv


def send_email(subject, body, to):
    s = smtplib.SMTP("localhost")
    mime = MIMEText(body)
    mime["Subject"] = subject
    mime["To"] = to
    s.sendmail("detectron", to, mime.as_string())


def humanize_time_delta(sec):
    """Humanize timedelta given in seconds
    Args:
        sec (float): time difference in seconds. Must be positive.
    Returns:
        str - time difference as a readable string
    Example:
    .. code-block:: python
        print(humanize_time_delta(1))                                   # 1 second
        print(humanize_time_delta(60 + 1))                              # 1 minute 1 second
        print(humanize_time_delta(87.6))                                # 1 minute 27 seconds
        print(humanize_time_delta(0.01))                                # 0.01 seconds
        print(humanize_time_delta(60 * 60 + 1))                         # 1 hour 1 second
        print(humanize_time_delta(60 * 60 * 24 + 1))                    # 1 day 1 second
        print(humanize_time_delta(60 * 60 * 24 + 60 * 2 + 60*60*9 + 3)) # 1 day 9 hours 2 minutes 3 seconds
    """
    if sec < 0:
        logger.warning("humanize_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    if sec == 0:
        return "0 second"
    _time = datetime(2000, 1, 1) + timedelta(seconds=int(sec))
    units = ["day", "hour", "minute", "second"]
    vals = [int(sec // 86400), _time.hour, _time.minute, _time.second]
    if sec < 60:
        vals[-1] = sec

    def _format(v, u):
        return "{:.3g} {}{}".format(v, u, "s" if v > 1 else "")

    ans = []
    for v, u in zip(vals, units):
        if v > 0:
            ans.append(_format(v, u))
    return " ".join(ans)


@contextmanager
def change_env(name, val):
    """
    Args:
        name(str), val(str):
    Returns:
        a context where the environment variable ``name`` being set to
        ``val``. It will be set back after the context exits.
    """
    oldval = os.environ.get(name, None)
    os.environ[name] = val
    yield
    if oldval is None:
        del os.environ[name]
    else:
        os.environ[name] = oldval


_RNG_SEED = None


def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed within tensorpack.
    Args:
        seed (int):
    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.
    Example:
        Fix random seed in both tensorpack and tensorflow.
    .. code-block:: python
            import tensorpack.utils.utils as utils
            seed = 42
            utils.fix_rng_seed(seed)
            tesnorflow.set_random_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    Each called in the code to this function is guaranteed to return True the
    first time and False afterwards.
    Returns:
        bool: whether this is the first time this function gets called from this line of code.
    Example:
        .. code-block:: python
            if execute_only_once():
                # do something only once
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True


def _pick_tqdm_interval(file):
    # Heuristics to pick a update interval for progress bar that's nice-looking for users.
    isatty = file.isatty()
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream

        if isinstance(file, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        return 0.5
    else:
        # When run under mpirun/slurm, isatty is always False.
        # Here we apply some hacky heuristics for slurm.
        if "SLURM_JOB_ID" in os.environ:
            if int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) > 1:
                # multi-machine job, probably not interactive
                return 60
            else:
                # possibly interactive, so let's be conservative
                return 15

        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            if int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
                return 60

        # If not a tty, don't refresh progress bar that often
        return 180


def get_tqdm_kwargs(**kwargs):
    """Return default arguments to be used with tqdm.

    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]",
    )

    try:
        # Use this env var to override the refresh interval setting
        interval = float(os.environ["TENSORPACK_PROGRESS_REFRESH"])
    except KeyError:
        interval = _pick_tqdm_interval(kwargs.get("file", sys.stderr))

    default["mininterval"] = interval
    default.update(kwargs)
    return default


def get_tqdm(*args, **kwargs):
    """Similar to :func:`tqdm.tqdm()`, but use tensorpack's default options to
    have consistent style."""
    return tqdm(*args, **get_tqdm_kwargs(**kwargs))


def is_picklable(obj):
    try:
        pickle.dumps(obj)

    except pickle.PicklingError:
        return False
    return True

def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def normalize_to_255(img):
    if img.max() != img.min():
        res_img = (img - img.min()) / (img.max() - img.min())
        return res_img * 255
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(show_emb)
    return show_emb


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.

    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = a[:, :, 1] * b[:, :, 0] + a[:, :, 2] * b[:, :, 3] - a[:, :, 3] * b[:, :, 2] + a[:, :, 0] * b[:, :, 1]
    y = -a[:, :, 1] * b[:, :, 3] + a[:, :, 2] * b[:, :, 0] + a[:, :, 3] * b[:, :, 1] + a[:, :, 0] * b[:, :, 2]
    z = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] + a[:, :, 0] * b[:, :, 3]
    w = -a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 3] + a[:, :, 0] * b[:, :, 0]

    return torch.stack((w, x, y, z), dim=2)


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[:, 0]  # Remove added Corner dimension here.
    return q_ego


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """
    Args:
        translation: Nx3
        rot_allo: Nx3x3
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego

def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [
            1.0 - (yY + zZ),
            xY - wZ,
            xZ + wY,
            xY + wZ,
            1.0 - (xX + zZ),
            yZ - wX,
            xZ - wY,
            yZ + wX,
            1.0 - (xX + yY),
        ],
        dim=1,
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat