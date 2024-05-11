
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform, euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


def reset_to_motion_init(env: BaseEnv, env_ids: torch.Tensor, asset_name='c1', root_only=False):
    """Reset the scene to the default state specified in the scene configuration."""
    asset = env.scene[asset_name]
    ref_motion = env.motion_data.data[asset_name+'_ori_motion'][env.cur_ref_idx[env_ids]]

    default_root_state = ref_motion[:,66:66+13].clone()
    

    if asset.data.joint_pos.shape[-1]==63:
        default_root_state[:, 2] += 0.1 * torch.ones_like(default_root_state[:, 2])
        default_root_state[:,2] = default_root_state[:,2] * env.char_property[asset_name][1] 
    else:
        default_root_state[:,2] = asset.data.default_root_state[env_ids,2]

    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    
    r,p,y = euler_xyz_from_quat(default_root_state[:,3:7])
    head_quat = quat_from_euler_xyz(roll=0*r, pitch=0*p, yaw=y)
    default_root_state[:,3:7] = head_quat
    
    #### set into the physics simulation
    asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    
    
    # #### obtain default joint positions
    if asset.data.joint_pos.shape[-1]==63:
        default_joint_pos = ref_motion[:,79:79+63].clone() #+ 0*asset.data.default_joint_pos[env_ids]
        default_joint_vel = ref_motion[:,79+63:79+126].clone()*0
        
        #### set into the physics simulation
        asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
 
    asset.update(dt=env.physics_dt)


    
    
def sample_reference(env: BaseEnv, env_ids: torch.Tensor):
    env.cur_ref_idx[env_ids] = env.motion_data.sample(sample_size = len(env_ids))
    

    
    