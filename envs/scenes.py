import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.sim as sim_utils


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.orbit.terrains import TerrainImporterCfg

from transmimicV2_interaction.envs.assets import *





@configclass
class SmplHumanSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                physics_material=sim_utils.RigidBodyMaterialCfg(
                                friction_combine_mode="multiply",
                                restitution_combine_mode="multiply",
                                static_friction=1.0,
                                dynamic_friction=1.0,
                            ),
            )


    # articulation
    c1: ArticulationCfg = SMPLHUMAN_CFG.replace(prim_path="{ENV_REGEX_NS}/c1")
    contact_forces_c1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/c1/.*", history_length=3,  track_air_time=True)
    
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    


@configclass
class ChildSceneCfg(SmplHumanSceneCfg):
    c1: ArticulationCfg = CHILD_CFG.replace(prim_path="{ENV_REGEX_NS}/c1")
    c1.init_state.pos = (0.0, 0., .5)


@configclass
class Go2ArSceneCfg(SmplHumanSceneCfg):
    c1: ArticulationCfg = GO2ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/c1")
    c1.init_state.pos = (0.0, 0., 0.4)


@configclass
class HumanXHumanSceneCfg(SmplHumanSceneCfg):
    c2: ArticulationCfg = SMPLHUMAN_CFG.replace(prim_path="{ENV_REGEX_NS}/c2")
    c2.init_state.pos = (0.0, 1., 1.)
    contact_forces_c2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/c2/.*", history_length=3, track_air_time=True)


@configclass
class HumanXChildSceneCfg(HumanXHumanSceneCfg):
    c2: ArticulationCfg = CHILD_CFG.replace(prim_path="{ENV_REGEX_NS}/c2")
    c2.init_state.pos = (0.0, 1., 1.)


@configclass
class HumanXGo2ArSceneCfg(HumanXHumanSceneCfg):
    c2: ArticulationCfg = GO2ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/c2")
    c2.init_state.pos = (0.0, 1., .4)


@configclass
class ChildXChildSceneCfg(ChildSceneCfg):
    c2: ArticulationCfg = CHILD_CFG.replace(prim_path="{ENV_REGEX_NS}/c2")
    c2.init_state.pos = (0.0, 1., .5)


@configclass
class ChildXGo2ArSceneCfg(ChildSceneCfg):
    c2: ArticulationCfg = GO2ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/c2")
    c2.init_state.pos = (0.0, 1., .4)


@configclass
class Go2ArXGo2ArSceneCfg(HumanXGo2ArSceneCfg):
    c1: ArticulationCfg = GO2ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/c1")
    c1.init_state.pos = (0.0, 0., .4)