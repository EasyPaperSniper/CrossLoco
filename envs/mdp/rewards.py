import torch
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_error_magnitude, wrap_to_pi, quat_mul
from transmimicV2_interaction.utils.process_intergen.interGen_param import *
from transmimicV2_interaction.envs.mdp.observations import get_cur_AIG, get_cur_ref_motion, get_cur_ref_AIG


def energy_penlty(env, asset_name='c1'):
    """Energy penalty."""
    asset = env.scene[asset_name]
    return torch.sum(torch.abs(asset.data.joint_vel * asset.data.applied_torque),dim=1)


def tracking_reward(env, asset_name='c1'):
     
    # root pos tracking
    asset: Articulation = env.scene[asset_name]
    ref_motion = get_cur_ref_motion(env, asset_name)
    ref_joint_w_pos = ref_motion[:, 0, 0:66].reshape(env.num_envs,-1,3)
    cur_body_pos = (asset.data.body_state_w[:,SMPL_KEY_LINK_IDX,0:3]- env.scene.env_origins[:,0:3].unsqueeze(1))
    delta_body_pos = torch.norm(ref_joint_w_pos-cur_body_pos, dim=-1) 
    delta_root_ori = quat_error_magnitude(ref_motion[:, 0, 66+3:66+7], asset.data.root_quat_w)
    
    
    joint_pos_w_error = torch.exp(-torch.mean(delta_body_pos[:,SMPL_KEY_LINK_IDX_wo_foot],dim=1)* 1)
    # joint_pos_w_error = torch.exp(-torch.mean(delta_body_pos,dim=1)* 1)
    root_ori_error = torch.exp(-delta_root_ori * 5)
    delta_joint_pos = torch.mean(torch.abs( ref_motion[:, 0, 79:79+63]-asset.data.joint_pos), dim=-1)
    joint_pos_error =torch.exp(-delta_joint_pos*1)

    return     joint_pos_w_error*joint_pos_error # + root_ori_error  
            


def root_height_tracking(env, asset_name='c1', tgt_height=0.3):
    asset: Articulation = env.scene[asset_name]
    ref_root_height = tgt_height
    cur_root_height = asset.data.root_pos_w[:,2]- env.scene.env_origins[:,2]

    return torch.exp(-10*torch.abs( ref_root_height-cur_root_height))#*torch.exp(-1*torch.norm(ori, dim=-1))



def root_ori_tracking(env, asset_name='c1'):
    asset: Articulation = env.scene[asset_name]
    # ori = asset.data.projected_gravity_b[:, :2]
    ref_motion = get_cur_ref_motion(env, asset_name)
    delta_root_ori = quat_error_magnitude(ref_motion[:, 0, 66+3:66+7], asset.data.root_quat_w)
    return torch.exp(-delta_root_ori * 5)


def flat_orientation_l2(env, std: float, asset_name) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_name]
    return  torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)/std**2


def track_joint_pos_target_l2(env, target: float, asset_cfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    # pos_error = torch.sum(torch.square(target - joint_pos), dim=1)
    # return torch.exp(-pos_error)
    return torch.sum(torch.square(joint_pos - target), dim=1)

def feet_air_time(env, sensor_cfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward

def AIG_reward(env):
    ref_AIG = get_cur_ref_AIG(env) 
    cur_AIG = get_cur_AIG(env)
    group_factor = (env.char_property['c1'][1]+env.char_property['c2'][1])/2
    height_factor = (env.char_property['c1'][2]+env.char_property['c2'][2])/2
    norm_ref_AIG = ref_AIG[:,0] * group_factor
    
    
    min_ref_AIG = torch.min(torch.norm(norm_ref_AIG[:,:,0:3],dim=-1),dim=-1)
    min_norm_ref_AIG = min_ref_AIG.values 
    min_norm_ref_AIG_idx = min_ref_AIG.indices
    delta_AIG = torch.abs(min_norm_ref_AIG - torch.norm(cur_AIG[:,:,0:3], dim=-1)[:,-1])/torch.norm(norm_ref_AIG[:,0,0:3], dim=-1)
    
    # weighted_length_AIG = torch.exp(-10*torch.norm(ref_AIG[:,0], dim=-1)/torch.sum(torch.norm(ref_AIG[:,0], dim=-1), dim=-1, keepdim=True))
    # AIG_length_reward = torch.exp(-torch.mean(weighted_length_AIG*delta_length_AIG, dim=1) * 10)
    delta_AIG_reward = torch.exp(-5*delta_AIG)

    ref_AIG_height = norm_ref_AIG[torch.arange(env.num_envs),min_norm_ref_AIG_idx,-1]/group_factor*height_factor
    delta_AIG_height = torch.abs(ref_AIG_height.unsqueeze(-1) - cur_AIG[:,:,-1])[:,-1]/height_factor
    delta_AIG_height_reward = torch.exp(-5* delta_AIG_height)
    

    delta_root_AIG = torch.norm(norm_ref_AIG[:,0,0:2] - cur_AIG[:,0, 0:2], dim=-1)/torch.norm(norm_ref_AIG[:,0,0:2], dim=-1)
    AIG_root_reward = torch.exp(-1.5*delta_root_AIG)
    AIG_root_reward_rough = torch.exp(-0.5*delta_root_AIG)


    
    return delta_AIG_height_reward*delta_AIG_reward * AIG_root_reward + 0.2* AIG_root_reward_rough 




def Xmorph_reward(env):
    if env.Xmorph_mapper is None:
        return torch.zeros((env.num_envs), device=env.device)
    scr_pose, tgt_pose = env.get_Xmorph_state()
    reconst_loss = 0.5 * torch.norm(env.Xmorph_mapper.forward(scr_pose) - tgt_pose, dim=-1) +  \
                        0.5 * torch.norm(env.Xmorph_mapper.inverse(tgt_pose) - scr_pose, dim=-1)

    return torch.exp(-1* reconst_loss)