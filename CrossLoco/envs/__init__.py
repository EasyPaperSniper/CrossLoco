from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
from CrossLoco.envs.aliengoZ1.aliengoZ1_config import aliengoZ1RoughCfg, aliengoZ1RoughCfgPPO
from CrossLoco.envs.aliengoZ1.aliengoZ1 import aliengoZ1_Base
from CrossLoco.envs.aliengo.aliengo_config import aliengoRoughCfg, aliengoRoughCfgPPO
from CrossLoco.envs.aliengo.aliengo import aliengo_Base
from CrossLoco.envs.B1Z1.B1Z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO
from CrossLoco.envs.B1Z1.B1Z1 import B1Z1
from CrossLoco.envs.crossloco.crossloco_aliengoZ1 import aliengoZ1_crossloco
from CrossLoco.envs.crossloco.crossloco_aliengoZ1_config import aliengoZ1crosslocoCfg, aliengoZ1crossloco_learning
from CrossLoco.envs.crossloco.crossloco_aliengo_config import aliengocrosslocoCfg, aliengocrossloco_learning

import os
from CrossLoco.utils.task_registry import crossloco_registry




crossloco_registry.register( "aliengoZ1", aliengoZ1_Base, aliengoZ1RoughCfg(), aliengoZ1RoughCfgPPO() )
crossloco_registry.register( "aliengo", aliengo_Base, aliengoRoughCfg(), aliengoRoughCfgPPO() )
crossloco_registry.register( "B1Z1", B1Z1, B1Z1RoughCfg(), B1Z1RoughCfgPPO() )
crossloco_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
crossloco_registry.register( "crossloco_aliengoZ1", aliengoZ1_crossloco, aliengoZ1crosslocoCfg(), aliengoZ1crossloco_learning() )
crossloco_registry.register( "crossloco_aliengo", aliengoZ1_crossloco, aliengocrosslocoCfg(), aliengocrossloco_learning() )