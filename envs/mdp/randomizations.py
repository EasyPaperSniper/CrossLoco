
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform, euler_xyz_from_quat


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


def reset_to_ref(env: BaseEnv,env_ids: torch.Tensor,):
    
    env.cur_ref_idx[env_ids] = env.motion_data.sample(sample_size = len(env_ids))
    
    asset = env.scene['robot']
    ref_motion = env.motion_data.data[env.cur_ref_idx[env_ids]]
    default_root_state = ref_motion[:,0,0:13]
    

    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    
    r,p,y = euler_xyz_from_quat(default_root_state[:,3:7])
    head_quat = quat_from_euler_xyz(roll=0*r, pitch=0*p, yaw=y)
    default_root_state[:,3:7] = head_quat
    
    #### set into the physics simulation
    asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)