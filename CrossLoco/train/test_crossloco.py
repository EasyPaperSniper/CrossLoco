from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit
from CrossLoco.envs import *
from CrossLoco import crossloco_ROOT_DIR
from CrossLoco.utils.task_registry import crossloco_registry
from CrossLoco.utils import Logger

import numpy as np
import torch

import moviepy.editor as mpy


def play(args):
    env_cfg, train_cfg = crossloco_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = crossloco_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run ='initial_raise_hand'
    train_cfg.runner.checkpoint = '600'
    log_root = os.path.join(crossloco_ROOT_DIR, 'logs', args.task)
    crossloco_runner, train_cfg = crossloco_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=log_root)
    policy, h_reconst_net, r_reconst_net = crossloco_runner.get_inference_policy(device=env.device)
    
    

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = env.max_episode_length # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    ori_camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)+ np.array([-1.0, -0, 0.0])
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)


    img_idx = 0
    motion_idx = 0
    

    human_local_dataset = torch.from_numpy(np.load(log_root+'/'+train_cfg.runner.load_run+'/human_local_dataset.npy')).to(env.device, dtype=torch.float)
    human_root_dataset = torch.from_numpy(np.load(log_root+'/'+train_cfg.runner.load_run+'/human_root_dataset.npy')).to(env.device, dtype=torch.float)
    env.update_reconst_net(h_reconst_net, r_reconst_net)
    env.load_human_motion(human_local_dataset, human_root_dataset, motion_idx*np.ones(env.num_envs))
    obs, _ = env.reset()

    dataset_info = np.load(log_root+'/'+train_cfg.runner.load_run+'/dataset_info.npy')
    print(dataset_info[motion_idx])
    start_index = int(dataset_info[motion_idx][2])
    motion_length = int(dataset_info[motion_idx][3])

    if VIZ_HUMAN:
        from pybullet_motion_viewer.viewer.viewer_env import simple_viewer
        viewer = simple_viewer()
        viewer.load_bvh_motion([dataset_info[motion_idx][0]], dataset=['Lafan1'], clip=[[start_index,start_index+motion_length]])

    if SAVE_ROBOT_MOTION:
        import pybullet
        
        pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        pybullet.resetSimulation()
        py_robot = pybullet.loadURDF(
                        env_cfg.asset.file, env.root_states[0, 0:3].cpu().numpy(),
                        env.root_states[0, 3:7].cpu().numpy())
        py_ground = pybullet.loadURDF('./pybullet_visualizer/pybullet_motion_viewer/robots/urdf/plane/plane.urdf')
        # for j in range(pybullet.getNumJoints(py_robot)):
        #     jointInfo = pybullet.getJointInfo(py_robot, j)
        #     joint_name = jointInfo[1]
        #     joint_name = joint_name.decode("utf8")
        #     print(j, joint_name)
        MOTOR_IDX = [6,7,8,2,3,4,14,15,16,10,11,12,]

        
        rob_link_states,rob_link_states_blender = [],[]
        pybullet_video = []
    
    for i in range(int(env.max_episode_length)*10):
        
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        root_pos = env.root_states[robot_index].detach().cpu().numpy()[:3]

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(crossloco_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            
            camera_direction = - np.array([0, -2, 0.2])
            camera_position = root_pos - camera_direction
            env.set_camera(camera_position, camera_position + camera_direction)
            # print('robot', env.rob_heading_quat[0], env.root_states[0, 3:7])
            # print('rob_heading', env.rob_heading[0], 'gravity', env.gravity_vec[0])     
            # print( 'human', env.human_root_traj[0, env.episode_length_buf-1,3:]) 
        if VIZ_HUMAN:
            human_pic, h_root_pos = viewer.step(i%env.max_episode_length, save_video=False)


        



if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    VIZ_HUMAN =False
    SAVE_ROBOT_MOTION=True
    args = get_args()
    play(args)