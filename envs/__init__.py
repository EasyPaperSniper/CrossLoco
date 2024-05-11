import gymnasium as gym

from transmimicV2_interaction.envs.env_cfg import *
from transmimicV2_interaction.envs.train_cfg import *
from transmimicV2_interaction.envs.transmimicV2_env import *


gym.register(
    id="Human_human",
    entry_point="transmimicV2_interaction.envs.transmimicV2_env:TransMimicV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HumanXHumanEnvCfg,
        "rl_cfg_entry_point": TM2PPORunnerCfg,
    },
)



gym.register(
    id="Go2Ar_Go2Ar",
    entry_point="transmimicV2_interaction.envs.transmimicV2_env:TransMimicV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2ArXGo2ArEnvCfg,
        "rl_cfg_entry_point": TM2PPORunnerCfg,
    },
)
