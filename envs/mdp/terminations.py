import torch
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_error_magnitude
from transmimicV2_interaction.utils.process_intergen.interGen_param import *
from transmimicV2_interaction.envs.mdp.observations import get_cur_AIG, get_cur_ref_AIG, get_cur_ref_motion


def root_off_track_z(env, asset_name, threshold: float=0.4):
    asset: Articulation = env.scene[asset_name]
    cur_root_pos = asset.data.root_pos_w[:, 0:3] - env.scene.env_origins[:,0:3]
    return (cur_root_pos[:,2] < threshold) * (env.episode_length_buf>0)

def root_off_track_ori(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion =  get_cur_ref_motion(env, asset_name)
    root_ori_error = torch.mean(torch.abs(ref_motion[:, 0, 66+3:66+7]- asset.data.root_quat_w), dim=-1)
    return (root_ori_error > threshold) * (env.episode_length_buf>0)

def root_off_track_xy(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion =  get_cur_ref_motion(env, asset_name)
    cur_root_pos = asset.data.root_pos_w[:, 0:3] - env.scene.env_origins[:,0:3]
    root_pos_error = torch.sum(torch.abs(ref_motion[:, 0, 66:66+2]- cur_root_pos[:,0:2]), dim=-1)
    return (root_pos_error > threshold) * (env.episode_length_buf>15)

def joint_off_track(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion = get_cur_ref_motion(env, asset_name)   
    joint_error =  torch.abs(ref_motion[:,0, 79:79+63]- asset.data.joint_pos)
    return (torch.sum(joint_error > threshold, dim=-1)>0) * (env.episode_length_buf>15)


def joint_pos_w_off_track(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion = get_cur_ref_motion(env, asset_name)
    ref_joint_w_pos = ref_motion[:, 0, 0:66].reshape(env.num_envs,-1,3)
    # cur_root_pos = asset.data.root_pos_w[:, 0:3] - env.scene.env_origins[:,0:3]
    cur_body_pos = (asset.data.body_state_w[:,SMPL_KEY_LINK_IDX,0:3]- env.scene.env_origins[:,0:3].unsqueeze(1))
    delta_body_pos = torch.norm(ref_joint_w_pos-cur_body_pos, dim=-1) 
    return (torch.mean(delta_body_pos[:, SMPL_KEY_LINK_IDX_wo_limb],dim=-1)>threshold)* (env.episode_length_buf>30)


def joint_pos_w_off_track_max(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion =  get_cur_ref_motion(env, asset_name)
    ref_joint_w_pos = ref_motion[:, 0, 0:66].reshape(env.num_envs,-1,3)
    # cur_root_pos = asset.data.root_pos_w[:, 0:3] - env.scene.env_origins[:,0:3]
    cur_body_pos = (asset.data.body_state_w[:,SMPL_KEY_LINK_IDX,0:3]- env.scene.env_origins[:,0:3].unsqueeze(1))
    delta_body_pos = torch.norm(ref_joint_w_pos-cur_body_pos, dim=-1) 
    return (torch.sum(delta_body_pos[:,SMPL_KEY_LINK_IDX_wo_limb ]>threshold,dim=1)>0)* (env.episode_length_buf>30)


def root_ori_w_off_max(env, asset_name, threshold: float=0.5):
    asset: Articulation = env.scene[asset_name]
    ref_motion =  get_cur_ref_motion(env, asset_name)
    delta_root_ori = quat_error_magnitude(ref_motion[:, 0, 66+3:66+7], asset.data.root_quat_w)
    
    return (delta_root_ori>threshold) * (env.episode_length_buf>30)


def AIG_off_max(env, threshold):
    ref_AIG = get_cur_ref_AIG(env)
    cur_AIG = get_cur_AIG(env)
    # delta_AIG_norm = torch.abs(torch.norm(ref_AIG[:,0], dim=-1) -  torch.norm(cur_AIG, dim=-1))/torch.norm(ref_AIG[:,0,0], dim=-1).unsqueeze(-1)
    # delta_AIG_norm = torch.norm(ref_AIG[:,0] - cur_AIG, dim=-1)/torch.norm(ref_AIG[:,0,0], dim=-1).unsqueeze(-1)
    norm_min_ref_AIG = torch.min(torch.norm(ref_AIG[:,0,:,0:3], dim=-1),dim=-1).values * (env.char_property['c1'][1]+env.char_property['c2'][1])/2
    delta_AIG_norm = torch.abs(norm_min_ref_AIG - torch.norm(cur_AIG[:,:,0:3], dim=-1)[:,-1])/torch.norm(ref_AIG[:,0,0,0:3], dim=-1)
    return (delta_AIG_norm > threshold) * (env.episode_length_buf > 90)

def AIG_center_off(env, threshold):
    ref_AIG = get_cur_ref_AIG(env) 
    cur_AIG = get_cur_AIG(env)
    group_factor = (env.char_property['c1'][1]+env.char_property['c2'][1])/2
    height_factor = (env.char_property['c1'][2]+env.char_property['c2'][2])/2
    norm_ref_AIG = ref_AIG[:,0] * group_factor
    delta_root_AIG = torch.norm(norm_ref_AIG[:,0,0:2] - cur_AIG[:,0, 0:2], dim=-1)/torch.norm(norm_ref_AIG[:,0,0:2], dim=-1)
    
    return (delta_root_AIG>threshold)* (env.episode_length_buf > 90)
    