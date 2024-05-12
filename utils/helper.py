import os
import copy
import torch
import numpy as np
import random
from datetime import datetime

import argparse
from typing import TYPE_CHECKING, Any
import gymnasium as gym


from CrossLoco import CrossLoco_ROOT_DIR


def launch_isaac_get_args():
    from omni.isaac.orbit.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser(description="CrossLoco")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")

    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument("--physics_gpu", type=int, default='0', help="Use CPU pipeline.")
    parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

    add_train_args(parser)
    add_crossloco_args(parser)


    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()


    # launch the simulator
    # load cheaper kit config in headless
    if args_cli.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    # launch omniverse app
    app_launcher = AppLauncher(args_cli, experience=app_experience)
    simulation_app = app_launcher.app
    
    
    return simulation_app, args_cli


def add_crossloco_args(parser: argparse.ArgumentParser):
    return


def add_train_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
        
    
def parse_env_cfg(args_cli) -> dict | Any:
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        use_gpu: Whether to use GPU/CPU pipeline. Defaults to None, in which case it is left unchanged.
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object. This is either a dictionary or a class object.

    Raises:
        ValueError: If the task name is not provided, i.e. None.
    """
    from omni.isaac.orbit.utils import update_class_from_dict, update_dict
    from omni.isaac.orbit_tasks.utils import load_cfg_from_registry
    
    
    task_name = args_cli.task
    use_gpu = not args_cli.cpu
    num_envs = args_cli.num_envs 
    use_fabric = not args_cli.disable_fabric
    
    
    # check if a task name is provided
    if task_name is None:
        raise ValueError("Please provide a valid task name. Hint: Use --task <task_name>.")
    # create a dictionary to update from
    args_cfg = {"sim": {"physx": dict()}, "scene": dict()}
    # resolve pipeline to use (based on input)
    if use_gpu is not None:
        if not use_gpu:
            args_cfg["sim"]["use_gpu_pipeline"] = False
            args_cfg["sim"]["physx"]["use_gpu"] = False
            args_cfg["sim"]["device"] = "cpu"
        else:
            args_cfg["sim"]["use_gpu_pipeline"] = True
            args_cfg["sim"]["physx"]["use_gpu"] = True
            args_cfg["sim"]["device"] = "cuda:{}".format(args_cli.physics_gpu)

    # number of environments
    if num_envs is not None:
        args_cfg["scene"]["num_envs"] = num_envs

    # load the default configuration
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    # update the main configuration
    if isinstance(cfg, dict):
        cfg = update_dict(cfg, args_cfg)
    else:
        update_class_from_dict(cfg, args_cfg)

    return cfg
    
    
def parse_agent_cfg(args_cli: argparse.Namespace) :
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """

    # load the default configuration
    from omni.isaac.orbit_tasks.utils import  load_cfg_from_registry
    
    rslrl_cfg = load_cfg_from_registry(args_cli.task, "rl_cfg_entry_point")

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        rslrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rslrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        rslrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        rslrl_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        rslrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rslrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rslrl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rslrl_cfg.wandb_project = args_cli.log_project_name
        rslrl_cfg.neptune_project = args_cli.log_project_name

    return rslrl_cfg



def get_log_dir(env_cfg, agent_cfg, args, log_root=None):
    # specify directory for logging experiments    
    if log_root is None:
        log_root_path = os.path.join(CrossLoco_ROOT_DIR, 'results', args.task, agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if not agent_cfg.run_name:
        agent_cfg.run_name = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(log_root_path, agent_cfg.run_name)

    return log_root_path, log_dir



def dump_cfg(env_cfg, agent_cfg, log_dir):
    from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml
    
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)


def set_seed(seed, env):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_env(env_cfg, args_cli, log_dir):
    from omni.isaac.orbit.utils.dict import print_dict 
    from CrossLoco.envs.crossloco_env import CrossLocoVecEnvWrapper
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = CrossLocoVecEnvWrapper(env)
    return env


def load_runner_run(runner, log_root_path, agent_cfg):
    from omni.isaac.orbit_tasks.utils import get_checkpoint_path
    
    # get path to previous checkpoint
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner.load(resume_path)
    return resume_path