from typing import Sequence

import numpy as np
import torch
from typing import Any

from omni.isaac.orbit.envs.base_env import VecEnvObs
from omni.isaac.orbit.envs.rl_task_env_cfg import RLTaskEnvCfg
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from CrossLoco.envs.scenes import *
from CrossLoco.envs.env_cfg import *
from CrossLoco.utils.human_dataset import CrossLoco_dataset


class CrossLocoBaseEnv(RLTaskEnv):

    def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None, **kwargs):
        self.motion_data = CrossLoco_dataset(motion_dir=cfg.ref_motion['motion_dir'], 
                                     motion_length= cfg.ref_motion['motion_length'],
                                     ref_horizon=cfg.ref_motion['ref_horizon'],
                                     device=cfg.sim.device)
        self.cur_ref_idx =  torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.h2r_mapper, self.r2h_mapper = None, None

        super().__init__(cfg, render_mode, **kwargs)
        
        self.metadata['render_fps'] = int(1/self.step_dt)   
        self.cfg.episode_length_s = self.motion_data.motion_length * self.step_dt
        self.num_obs = sum(value[0] for value in self.observation_manager.group_obs_dim.values())
        self.num_action = self.action_manager.action_term_dim[0]
        
        h_pose, r_pose = self.get_X_pose()
        self.h_pose_dim = h_pose.shape[-1]
        self.r_pose_dim = r_pose.shape[-1]
        
    def get_X_pose(self,):
        h_pos = self.motion_data.get_cur_ref_pose(self.cur_ref_idx, self.episode_length_buf )
        r_pos = self.scene['robot'].data.joint_pos
        return h_pos, r_pos
    
    
    def update_mappers(self, h2r_mapper, r2h_mapper):
        self.h2r_mapper = h2r_mapper
        self.r2h_mapper = r2h_mapper

        

        


class CrossLocoVecEnvWrapper(RslRlVecEnvWrapper):
    """Wraps around Orbit environment for RSL-RL library

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: CrossLocoBaseEnv):
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
        self.num_actions = self.unwrapped.num_action
        self.num_obs = self.unwrapped.num_obs
        self.h_pose_dim = self.unwrapped.h_pose_dim
        self.r_pose_dim = self.unwrapped.r_pose_dim
             
        self.env.reset()


    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        obs = torch.cat([obs_dict[key] for key in obs_dict.keys()], dim=1).to(self.device)
        return obs, {"observations": obs_dict}


    """
    Operations - MDP
    """

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
    
    
    """
    Cross Loco
    """
    def get_X_pose(self):
        h_pos, r_pos = self.unwrapped.get_X_pose()
        return h_pos, r_pos
    
    def update_mappers(self,h2r_mapper, r2h_mapper):
        self.unwrapped.h2r_mapper = h2r_mapper
        self.unwrapped.r2h_mapper = r2h_mapper
