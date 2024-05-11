from typing import Sequence, Any
import numpy as np
import torch


import gymnasium as gym
from rsl_rl.env import VecEnv
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.actuators import  ImplicitActuator

from transmimicV2_interaction.envs.env_cfg import RLTaskEnvCfg
from transmimicV2_interaction.envs.scenes import *
from transmimicV2_interaction.envs.env_cfg import *

from transmimicV2_interaction.utils.process_intergen.interGen_param import *
from transmimicV2_interaction.utils.process_intergen.gen_dataset import TM2_phyTrain_dataset



class TransMimicV2Env(RLTaskEnv):
    def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None,  **kwargs):
        
        self.motion_data = TM2_phyTrain_dataset(motion_name=cfg.ref_motion['motion_name'], 
                                     motion_length= cfg.ref_motion['motion_length'],
                                     ref_horizon=cfg.ref_motion['ref_horizon'],
                                     device=cfg.sim.device)
        self.cur_ref_idx =  torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.cur_ref_input = {'c1': None, 'c2': None}
        self.char_property = cfg.char_property
        self.Xmorph_mapper = None
        self.aig_idx = torch.tensor([0, 20, 21, 20* 22, 21*22, 21*22-2, 21*22-1, 22*22-2,22*22-1]).to(device=cfg.sim.device)
        # self.aig_idx = torch.tensor([0, 22, 22, 23* 23-1, 23* 23-1,23* 23-1, 23* 23-1, 23*23-1,23*23-1]).to(device=cfg.sim.device)
        
        
        super().__init__(cfg, render_mode, **kwargs)


        self.cfg.episode_length_s = self.motion_data.motion_length * self.step_dt     
        self.metadata['render_fps'] = int(1/self.step_dt)   
        
        self.num_obs_list = []
        for keys, value in self.observation_manager.group_obs_dim.items():
            self.num_obs_list.append(value[0])
        self.num_action_list = self.action_manager.action_term_dim

        pos_limit = self.scene['c1'].root_physx_view.get_dof_limits()[0].to(device=self.device)
        self.motion_data.data['c1_ori_motion'][:, 79:79+63] = torch.clamp(self.motion_data.data['c1_ori_motion'][:, 79:79+63], min=pos_limit[:,0], max=pos_limit[:,1])
        self.motion_data.data['c2_ori_motion'][:, 79:79+63] = torch.clamp(self.motion_data.data['c2_ori_motion'][:, 79:79+63], min=pos_limit[:,0], max=pos_limit[:,1])
        

    def update_Xmorph_mapper(self, Xmorph_mapper):
        self.Xmorph_mapper = Xmorph_mapper
        
    def get_Xmorph_state(self):
        tgt_pose = torch.cat([self.scene['c1'].data.joint_pos,self.scene['c2'].data.joint_pos], dim=-1)
        src_pose = torch.cat([ self.cur_ref_input['c1'][:,0, 79:79+63], self.cur_ref_input['c2'][:,0, 79:79+63]], dim=-1)
        return src_pose, tgt_pose
        
        
    def update_view_location(self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None):
        self.viewport_camera_controller.update_view_to_asset_root('c1')
        self.viewport_camera_controller.update_view_location(eye, lookat)
    

    def set_view_env_index(self, env_index: int):
        self.viewport_camera_controller.set_view_env_index(env_index)

    
    def set_reference_pos(self, c1_root, c1_joint, c2_root, c2_joint):
        tgt_c1_root = c1_root.clone()
        tgt_c2_root = c2_root.clone()
        
        tgt_c1_root[:,0:3] += self.scene.env_origins
        tgt_c2_root[:,0:3] += self.scene.env_origins
        self.scene['c1'].write_root_pose_to_sim(tgt_c1_root) 
        self.scene['c2'].write_root_pose_to_sim(tgt_c2_root)
        self.scene['c1'].write_root_velocity_to_sim(torch.zeros((6), device=self.device))
        self.scene['c2'].write_root_velocity_to_sim(torch.zeros((6), device=self.device))
        

        self.scene['c1'].write_joint_state_to_sim(position=c1_joint, velocity=torch.zeros_like(c1_joint))
        self.scene['c2'].write_joint_state_to_sim(position=c2_joint, velocity=torch.zeros_like(c2_joint))

        self.scene.write_data_to_sim()
        self.sim.step(render=False)    
        self.scene.update(dt=self.physics_dt)
        self.sim.render()
        
        return 


class TransMimicV2Toy(TransMimicV2Env):
    def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.sim.gravity = (0,0,-0.1)
        super().__init__(cfg, render_mode, **kwargs)

        
        
    def step(self, action):
        
        c1_ref_motion =  self.cur_ref_input['c1'][:,0]
        c2_ref_motion =  self.cur_ref_input['c2'][:,0]
        
        c1_root_pos = self.scene['c1'].data.root_state_w[:,0:7].clone()
        c2_root_pos = self.scene['c2'].data.root_state_w[:,0:7].clone()
        c1_heading = self.scene['c1'].data.heading_w
        c2_heading = self.scene['c2'].data.heading_w
        
        
        c1_root_pos[:,0:3] -= self.scene.env_origins
        c2_root_pos[:,0:3] -= self.scene.env_origins
        

        c1_cur_joint_pos = self.scene['c1'].data.joint_pos.clone()
        c2_cur_joint_pos = self.scene['c2'].data.joint_pos.clone()

        c1_joint_pos = self.scene['c1'].data.default_joint_pos.clone() # c1_ref_motion[:, 79:79+63].clone() # 
        c2_joint_pos =  self.scene['c2'].data.default_joint_pos.clone() # c2_ref_motion[:, 79:79+63].clone() # 
        
        c1_action = action[:, :c1_joint_pos.shape[-1]]
        c2_action = action[:, c1_joint_pos.shape[-1]:]
        
        ########### shit code warning ###############
        active_joint_idx ={'18':[12, 13, 14, 15, 16, 17],
                           '63':  [2,5,8,11,14,17,20,23,26,29,30,31,34,35,36] + list(range(39, 63))  
                           } 

        c1_root_pos[:,0:2] += 0.1*c1_action[:,0].unsqueeze(1) * \
                    torch.cat([torch.cos(c1_action[:,1]).unsqueeze(1), torch.sin(c1_action[:,1]).unsqueeze(1)], dim=-1).clone()
        c1_root_pos[:,2] += c1_action[:,2].clone()*0.1
        c1_root_pos[:,3:7] = c1_action[:,3:7]/torch.norm(c1_action[:,3:7], dim=-1, keepdim=True).clone()
        
        c2_root_pos[:,0:2] += 0.1*c2_action[:,0].unsqueeze(1) * \
                    torch.cat([torch.cos(c2_action[:,1]).unsqueeze(1), torch.sin(c2_action[:,1]).unsqueeze(1)], dim=-1).clone()
        c2_root_pos[:,2] += c2_action[:,2].clone()*0.1
        c2_root_pos[:,3:7] = c2_action[:,3:7]/torch.norm(c2_action[:,3:7], dim=-1, keepdim=True).clone()
        

        c1_pos_limit = self.scene['c1'].root_physx_view.get_dof_limits()[0].to(device=self.device)
        c2_pos_limit = self.scene['c2'].root_physx_view.get_dof_limits()[0].to(device=self.device)
        
        c1_joint_idx = active_joint_idx[str(c1_action.shape[-1])]
        c1_joint_pos[:,c1_joint_idx] = torch.clamp(c1_action[:,7:7+len(c1_joint_idx)]*0.1, min=c1_pos_limit[c1_joint_idx,0], max=c1_pos_limit[c1_joint_idx,1]).clone()
        
        c2_joint_idx = active_joint_idx[str(c2_action.shape[-1])]
        c2_joint_pos[:,c2_joint_idx] = torch.clamp(c2_action[:,7:7+len(c2_joint_idx)]*0.1, min=c2_pos_limit[c2_joint_idx,0], max=c2_pos_limit[c2_joint_idx,1]).clone()
        
        # c1_root_pos[:, 0:7] = c1_ref_motion[:, 66:66+7].clone()
        c1_root_pos[:, 0:2] = c1_ref_motion[:, 66:66+2].clone()
        c1_root_pos[:, 2] = 0.3#c1_ref_motion[:, 66:66+2].clone()
        # c1_joint_pos = c1_ref_motion[:, 79:79+63].clone()
        # c2_root_pos[:, 0:7] = c2_ref_motion[:, 66:69+4].clone() 
        c2_root_pos[:, 0:2] = c2_ref_motion[:, 66:66+2].clone()
        c2_root_pos[:, 2] = 0.3
        # c2_joint_pos = c2_ref_motion[:, 79:79+63].clone()
        
        self.set_reference_pos(c1_root_pos, c1_joint_pos, c2_root_pos, c2_joint_pos)
        obs_dict, rew, terminated, truncated, extras = super().step(torch.cat([c1_joint_pos, c2_joint_pos], dim=1).to(device=self.device))
        
        return obs_dict, rew, terminated, truncated, extras
    
    
    
    
class TM2VecEnvWrapper(VecEnv):
    """Wraps around Orbit environment for RSL-RL library

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: RLTaskEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`RLTaskEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, RLTaskEnv):
            raise ValueError(f"The environment must be inherited from RLTaskEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_action_list = self.unwrapped.num_action_list
        self.num_obs_list = self.unwrapped.num_obs_list

        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> RLTaskEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        obs = torch.cat([obs_dict[key] for key in obs_dict.keys()], dim=1).to(self.device)
        return obs, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        obs = torch.cat([obs_dict[key] for key in obs_dict.keys()], dim=1).to(self.device)
        return obs, {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = torch.cat([obs_dict[key] for key in obs_dict.keys()], dim=1).to(self.device)
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
    
    def get_Xmorph_state(self):
        src_pose, tgt_pose = self.env.get_Xmorph_state()
        return src_pose, tgt_pose
    
    
    def update_Xmorph_mapper(self, Xmorph_mapper):
        self.env.Xmorph_mapper = Xmorph_mapper