import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from CrossLoco.envs import *
from CrossLoco import crossloco_ROOT_DIR
from legged_gym.utils import get_args
from CrossLoco.utils.task_registry import crossloco_registry
import torch

def train(args):
    env_cfg, train_cfg = crossloco_registry.get_cfgs(name=args.task)
    env, env_cfg = crossloco_registry.make_env(name=args.task, args=args)

    train_cfg.runner.resume = False
    # train_cfg.runner.load_run ='walk_crossloco'
    # train_cfg.runner.checkpoint = '2000'
    # train_cfg.runner.load_run ='mdm_walk'
    # train_cfg.runner.checkpoint = '3500'


    log_root = os.path.join(crossloco_ROOT_DIR, 'logs', args.task)
    ppo_runner, train_cfg = crossloco_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=log_root, train_cfg=train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)