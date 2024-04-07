import os
from CrossLoco import crossloco_ROOT_DIR, crossloco_ENVS_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import math
import torch

class aliengoZ1RoughCfg( LeggedRobotCfg ):
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

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.25,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.25,  # [rad]

            'arm_joint1': 0.0,  # [rad],
            'arm_joint2': 0.0,  # [rad],
            'arm_joint3': 0.0,  # [rad],
            'arm_joint4': 0.0,  # [rad],
            'arm_joint5': 0.0,  # [rad],
            'arm_joint6': 0.0,  # [rad],
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.,
                     'RL_thigh':100.,
                     'RL_calf':100,
                     'RR_thigh':100.,
                     'RR_calf':100,
                     'FL_thigh':40.,
                     'FL_calf':40,
                     'FR_thigh':40.,
                     'FR_calf':40,
                     'arm': 20}  # [N*m/rad]
        damping = {'hip': 2.,
                     'RL_thigh':5.,
                     'RL_calf':5,
                     'RR_thigh':5.,
                     'RR_calf':5,
                     'FL_thigh':2.,
                     'FL_calf':2,
                     'FR_thigh':2.,
                     'FR_calf':2,
                     'arm': 5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False


    class asset( LeggedRobotCfg.asset ):
        file = crossloco_ROOT_DIR+ '/crossloco/envs/aliengoZ1/urdf/aliengoZ1.urdf'
        name = "aliengoZ1"
        foot_name = "foot"
        penalize_contacts_on = ["calf",  ]
        terminate_after_contacts_on = ["base", 'arm_link', 'hip', 'trunk','thigh']
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_history_buffer = 1
        num_observations = 66 * num_history_buffer
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 18

    class terrain(LeggedRobotCfg.terrain):
        measure_heights = False
        curriculum = False
        mesh_type = "plane"


    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        heading_command = False
        class ranges:
            lin_vel_x = [.2, 1.] # min max [m/s]
            lin_vel_y = [-0.0, 0.1]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]    # min max [rad/s]
            heading = [-1.14, 1.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = False
        push_interval_s = 2
        push_duration_s = 0.25


    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -1.0
            base_height = -1.0
            tracking_lin_vel = 5.0

    class viewer:
        ref_env = 0
        pos = [0, -2, .6]  # [m]
        lookat = [0., 0, 0.3]  # [m]
    
    class noise(LeggedRobotCfg.noise):
        add_noise=True

class aliengoZ1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01


    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_aliengoZ1'
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        policy_class_name = "ActorCritic"