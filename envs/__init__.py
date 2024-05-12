
import gymnasium as gym
from CrossLoco.envs.env_cfg import *
from CrossLoco.envs.train_cfg import *



gym.register(
    id="CrossLoco_Go2",
    entry_point="CrossLoco.envs.crossloco_env:CrossLocoBaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rl_cfg_entry_point": CrossLocoPPORunnerCfg,
    },
)



