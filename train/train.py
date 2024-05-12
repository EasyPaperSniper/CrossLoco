from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
from CrossLoco.utils.helper import launch_isaac_get_args
simulation_app, args_cli = launch_isaac_get_args()


import carb
import traceback
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


from CrossLoco.envs import *
from CrossLoco.runner.CrossLoco_Runners import CrossLoco_Runner
import CrossLoco.utils.helper as helper





def train():
    
    # parse configuration
    env_cfg = helper.parse_env_cfg(args_cli)
    agent_cfg = helper.parse_agent_cfg(args_cli)
    log_root_path, log_dir = helper.get_log_dir(env_cfg, agent_cfg, args_cli)
    
    # build env
    env = helper.build_env(env_cfg, args_cli, log_dir)
    
    # create runner from rsl-rl
    runner = CrossLoco_Runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    
    # save resume path before creating a new log_dir
    agent_cfg.resume = False
    agent_cfg.load_run = 'Apr28_17-24-04'
    agent_cfg.load_checkpoint = 'model_104000.pt'

    if agent_cfg.resume:
        helper.load_runner_run(runner, log_root_path, agent_cfg)
      

    # set seed of the environment
    helper.set_seed(agent_cfg.seed, env)

    # dump the configuration into log-directory
    helper.dump_cfg(env_cfg, agent_cfg, log_dir)
    

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()




if __name__ == "__main__":
    try:
        # run the main execution
        train()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()