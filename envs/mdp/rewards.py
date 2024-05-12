import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor, RayCaster
# if TYPE_CHECKING:
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_error_magnitude, wrap_to_pi, quat_mul



def energy_penlty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Energy penalty."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel * asset.data.applied_torque),dim=1)



def cycle_consist(env):
    if env.h2r_mapper is None:
        return torch.zeros(env.num_envs).to(device=env.device)
    
    h_pos, r_pos = env.get_X_pose()
    with torch.inference_mode():
        r2h_reconst = env.r2h_mapper(r_pos)
        h2r_reconst = env.h2r_mapper(r2h_reconst)
        return 0.5* torch.exp(-torch.norm(h_pos-r2h_reconst, dim=-1)*2)+ 0.5* torch.exp(-torch.norm(r_pos-h2r_reconst, dim=-1)*2)
   
    
def root_pos_track(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    root_pose = asset.data.root_pos_w
    root_pose[:,:2] -= env.scene.env_origins
    
    ref_root_pos = env.motion_data.get_cur_root_norm(env.cur_ref_idx, env.episode_length_buf)
    return torch.exp(-2* torch.norm(root_pose-ref_root_pos, dim=-1))
    


