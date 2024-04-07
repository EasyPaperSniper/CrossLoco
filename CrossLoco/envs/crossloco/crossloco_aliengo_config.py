import os
import math
import torch


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from CrossLoco import crossloco_ROOT_DIR, crossloco_ENVS_DIR
from CrossLoco.envs.aliengo.aliengo_config import aliengoRoughCfg

class aliengocrosslocoCfg(aliengoRoughCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_observations = 302
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        robot_reconst_dim = 12
        human_reconst_dim = 32
        episode_length_s = 10


    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        reconst_sigma = 1
        tracking_sigma = 1
        height_sigma = 0.1
        velocity_sigma = 1

        class scales( ):
           reconst = .6
           tracking = .2
           height = .2
           action_rate = -0.01
           torques = -0.00002
           dof_pos_limits = -5.0
           collision = -1.
           dof_acc = -2e-8
           dof_vel = -0.0002
           orientation = -0.02



    class asset( LeggedRobotCfg.asset ):
        file = crossloco_ROOT_DIR+ '/crossloco/envs/aliengo/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["calf",  ]
        terminate_after_contacts_on = ["base", 'arm_link', 'hip', 'trunk','thigh']
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter


    class human_motion:
        horizon = 5

    class noise(LeggedRobotCfg.noise):
        add_noise=False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class domain_rand:
        randomize_friction = False
        friction_range = [0.9, 1.1]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.


class aliengocrossloco_learning( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'crosslocoRunner'
    
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512,512, 512]
        critic_hidden_dims = [512,512, 512]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        reconst_net_dims = [256, 64,256]
        reconst_activation = 'relu'

        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 5e-3
        num_learning_epochs = 5
        num_mini_batches = 2# mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        

    class runner:
        policy_class_name = 'ActorCritic'
        reconst_class_name = 'MLP'
        algorithm_class_name = 'PPO'
        reconst_alg_class_name = 'CycLearning'
        num_steps_per_env = 24 # per iteration
        max_iterations =1500 # number of policy updates

        

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = 'initial_run'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt