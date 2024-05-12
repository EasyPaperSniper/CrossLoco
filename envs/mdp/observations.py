import torch

from omni.isaac.orbit.utils.math import combine_frame_transforms, compute_pose_error, transform_points
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit.envs.mdp as mdp

def get_ref_idx(env):
    return env.episode_length_buf.unsqueeze(-1) +\
            torch.arange(env.motion_data.ref_horizon).to(device=env.device).unsqueeze(0)

def human_input(env):
    ref_idx = get_ref_idx(env)
    cur_ref_input = env.motion_data.get_cur_ref(env.cur_ref_idx, env.episode_length_buf)
    return cur_ref_input.reshape(env.num_envs, -1)