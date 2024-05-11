# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


from omni.isaac.orbit.utils.math import combine_frame_transforms, compute_pose_error, transform_points
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit.envs.mdp as mdp

from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *
from transmimicV2_interaction.utils.process_intergen.interGen_param import *

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv
    from transmimicV2_interaction.envs.transmimicV2_env import TransMimicV2Env




'''
Reference Motion
'''

def get_ref_idx(env):
    return env.episode_length_buf.unsqueeze(-1)//2 +\
            env.cur_ref_idx.unsqueeze(-1) +\
            torch.tensor([0, 3, env.motion_data.ref_horizon], device=env.device).unsqueeze(0)


def get_cur_ref_motion(env, asset_name='c1'):
    ref_idx = get_ref_idx(env)
    env.cur_ref_input[asset_name] = env.motion_data.data[asset_name+'_ori_motion'][ref_idx]
    return env.motion_data.data[asset_name+'_ori_motion'][ref_idx]


def get_cur_ref_AIG(env):
    ref_idx = get_ref_idx(env)
    return env.motion_data.data['AIG'][ref_idx]



def get_ref_motion_b(env, asset_name='c1'):
    ref_motion = get_cur_ref_motion(env, asset_name)
    asset: RigidObject = env.scene[asset_name]

    root_quat_w = asset.data.root_quat_w
    ref_joint_w_pos = ref_motion[:, :, 0:66].reshape(env.num_envs,-1,22,3)
    cur_body_pos = (asset.data.body_state_w[:,SMPL_KEY_LINK_IDX,0:3]- env.scene.env_origins[:,0:3].unsqueeze(1))
    delta_joint_pos_w =ref_joint_w_pos-cur_body_pos.unsqueeze(1)
    delta_joint_pos_local = transform_points(points=delta_joint_pos_w.reshape(env.num_envs, -1,3), pos=None, quat=root_quat_w)

    # ref_joint_pos = ref_motion[:,0,  79:79+63]
    # return ref_joint_pos
    return delta_joint_pos_local.reshape(env.num_envs,-1)

def get_ref_motion_w(env, asset_name='c1'):
    ref_motion = get_cur_ref_motion(env, asset_name)
    asset: RigidObject = env.scene[asset_name]
    ref_joint_w_pos = ref_motion[:, :, 0:66].reshape(env.num_envs,-1,22,3)
  
    return ref_joint_w_pos.reshape(env.num_envs,-1)



def get_neu_heading(env):
    c1_root_w = env.scene['c1'].data.root_state_w[:,0:3]
    c2_root_w = env.scene['c2'].data.root_state_w[:,0:3]

    forward = torch.tensor([1., 0, 0], device=env.device).unsqueeze(0).repeat(c1_root_w.shape[0], 1)
    neu_heading_w =  c1_root_w[:,:3] - c2_root_w[:,:3]
    neu_heading_w[:, 2] = 0
    neu_heading_w = neu_heading_w/torch.norm(neu_heading_w, dim=-1, keepdim=True)
    neu_heading_quat = qbetween(forward, neu_heading_w )
    return neu_heading_quat


def get_cur_body_pos(env, asset_name='c1'):
    asset: RigidObject = env.scene[asset_name]
    cur_body_pos = (asset.data.body_state_w[:,SMPL_KEY_LINK_IDX,0:3]- env.scene.env_origins[:,0:3].unsqueeze(1))
    return cur_body_pos.reshape(env.num_envs,-1)



def get_cur_AIG(env):
    # need to change
    c1_cur_body_pos = env.scene['c1'].data.body_state_w[:,env.char_property['c1'][-1],0:3]
    c2_cur_body_pos = env.scene['c2'].data.body_state_w[:,env.char_property['c2'][-1],0:3]
    AIG_pos = (c1_cur_body_pos.unsqueeze(2) - c2_cur_body_pos.unsqueeze(1)).reshape(c1_cur_body_pos.shape[0],-1, 3)[:, env.aig_idx]
    AIG_center = ((c1_cur_body_pos.unsqueeze(2) + c2_cur_body_pos.unsqueeze(1))/2).reshape(c1_cur_body_pos.shape[0],-1, 3)[:, env.aig_idx]
    
    # neu_heading_quat = get_neu_heading(env)
    # AIG_local = qrot(qinv(neu_heading_quat).unsqueeze(1).repeat(1, AIG_pos.shape[1], 1), AIG_pos)
    return torch.cat([AIG_pos, AIG_center], dim=-1).to(device=env.device)
    