import os
from CrossLoco import crossloco_ROOT_DIR, crossloco_ENVS_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import math
import torch

class aliengoRoughCfg( LeggedRobotCfg ):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,  # [rad]
            'RL_hip_joint': 0.,  # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RR_hip_joint': 0.,  # [rad]

            'FL_thigh_joint': 0.6,  # [rad]
            'RL_thigh_joint': 0.6,  # [rad]
            'FR_thigh_joint': 0.6,  # [rad]
            'RR_thigh_joint': 0.6,  # [rad]

            'FL_calf_joint': -1.25,  # [rad]
            'RL_calf_joint': -1.25,  # [rad]
            'FR_calf_joint': -1.25,  # [rad]
            'RR_calf_joint': -1.25,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False


    class asset( LeggedRobotCfg.asset ):
        file = crossloco_ROOT_DIR+ '/crossloco/envs/aliengo/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["calf",  ]
        terminate_after_contacts_on = ["base", 'link', 'hip', 'trunk','thigh']
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_history_buffer = 3
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        measure_heights = False
        curriculum = False
        mesh_type = "plane"


    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        heading_command = False
        class ranges:
            lin_vel_x = [-1., 1.] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_interval_s = 2
        push_duration_s = 0.25


    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    class viewer:
        ref_env = 0
        pos = [0, -2, .6]  # [m]
        lookat = [0., 0, 0.3]  # [m]
    
    class noise(LeggedRobotCfg.noise):
        add_noise=True

class aliengoRoughCfgPPO( LeggedRobotCfgPPO ):


    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01


    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_aliengo'
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        policy_class_name = "ActorCritic"
