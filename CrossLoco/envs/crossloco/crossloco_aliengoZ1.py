import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.base_task import BaseTask
# from legged_gym.utils.terrain import Terrain
from CrossLoco.utils.terrain import Terrain
from legged_gym import LEGGED_GYM_ROOT_DIR
from CrossLoco.envs.aliengoZ1.aliengoZ1 import aliengoZ1_Base
from CrossLoco.utils import *




class aliengoZ1_crossloco(aliengoZ1_Base):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.rob_reconst_dim = self.cfg.env.robot_reconst_dim
        self.h_reconst_dim = self.cfg.env.human_reconst_dim


        self.rob_height = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.rob_heading_quat= torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        self.rob_distance = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.h_reconst_state = torch.zeros(self.num_envs, self.h_reconst_dim, device=self.device, dtype=torch.float)
        self.r_reconst_state = torch.zeros(self.num_envs, self.rob_reconst_dim, device=self.device, dtype=torch.float)
        self.base_init_state = self.base_init_state.repeat(self.num_envs, 1)
        self.fixed_sample = False


        
    def load_human_motion(self, human_local_state, human_root_state, data_index=None, **kwargs):
        self.max_episode_length = human_local_state.shape[1]
        self.num_motions = human_local_state.shape[0]
        self.max_episode_length_s = self.max_episode_length * self.dt
        self.human_local_state = human_local_state
        self.human_root_state = human_root_state
        if data_index is None:
            self.human_traj_index = torch.randint(0, self.num_motions, (self.num_envs,), device=self.device)
        else:
            self.human_traj_index = torch.from_numpy(data_index).to(self.device, dtype=torch.int64)
            self.fixed_sample = True
        self.human_local_traj = self.human_local_state[self.human_traj_index, :, :]
        self.human_root_traj = self.human_root_state[self.human_traj_index, :, :]
        self.base_init_state[:, 3:7] = self.human_root_traj[:,0,3:7]
        

    def _resample_traj(self, env_ids):
        if not self.fixed_sample:
            env_index = torch.randint(0, self.num_motions, () ,device=self.device)
        else:
            env_index = self.human_traj_index[env_ids]
        self.human_local_traj[env_ids] = self.human_local_state[env_index, :, :]
        self.human_root_traj[env_ids] = self.human_root_state[env_index, :, :]
        self.base_init_state[env_ids, 3:7] = self.human_root_traj[env_ids, 0, 3:7]


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        index = torch.arange(self.num_envs).to(self.device)
        tracking_delta = torch.sum(torch.square(self.rob_distance[:,0:2] - self.human_root_traj[index, self.episode_length_buf-1,0:2]/2.5), dim=1)
                                
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= (tracking_delta > 3.)






    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._resample_traj(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state[env_ids]
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state[env_ids]
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



    def get_human_state_in_rob_frame(self,rob_heading_quat, rob_distance, idx):
        human_local_state, human_traj = self.human_trajectory_operator.get_human_state_in_rob_frame(rob_heading_quat, rob_distance, idx)

        return human_local_state, human_traj



    def compute_observations(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.rob_heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.rob_heading_quat = quat_from_angle_axis(self.rob_heading, -self.gravity_vec)        

        index = torch.arange(self.num_envs).to(self.device)
        self.rob_height = self.root_states[:, 2].unsqueeze(1)
        self.rob_distance = self.root_states[:, :3] - self.env_origins[:]
        self.obs_buf = torch.cat((  
                                    self.human_local_traj[index, self.episode_length_buf-1,:],
                                    self.rob_height,
                                    self.projected_gravity,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        self.h_reconst_state = self.human_local_traj[index, self.episode_length_buf-1]
        self.r_reconst_state = torch.cat((
                                    self.rob_height,
                                    self.projected_gravity,
                                    # self.base_lin_vel * self.obs_scales.lin_vel,
                                    # self.base_ang_vel  * self.obs_scales.ang_vel,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    ),dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def get_reconst_states(self,):
        return self.r_reconst_state, self.h_reconst_state


    def update_reconst_net(self, h_reconst_net, r_reconst_net):
        self.h_reconst_net = h_reconst_net
        self.r_reconst_net = r_reconst_net


    def _reward_reconst(self,):
        h_prediction = self.h_reconst_net(self.r_reconst_state)
        r_prediction = self.r_reconst_net(h_prediction)
        h_reconst_error = torch.norm(h_prediction-self.h_reconst_state, dim=1)
        r_reconst_error = torch.norm(r_prediction-self.r_reconst_state, dim=1)
        reconst_error = (h_reconst_error+r_reconst_error)/2
        return torch.exp(-reconst_error/self.cfg.rewards.reconst_sigma)
    

    def _reward_tracking(self,):
        index = torch.arange(self.num_envs).to(self.device)
        tracking_delta = torch.sum(torch.square(self.rob_heading_quat - self.human_root_traj[index, self.episode_length_buf-1,3:]), dim=1) + \
                                torch.sum(torch.square(self.rob_distance[:,0:2] - self.human_root_traj[index, self.episode_length_buf-1,0:2]/2.5), dim=1) 
        return torch.exp(-tracking_delta/self.cfg.rewards.tracking_sigma)
    
    def _reward_height(self,):
        index = torch.arange(self.num_envs).to(self.device)
        height_error = torch.norm(self.rob_height-((self.human_local_traj[index, self.episode_length_buf-1,3:4]/0.9-1)*0.42+0.4), dim=1)
        return torch.exp(-height_error/self.cfg.rewards.height_sigma)
    

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    

