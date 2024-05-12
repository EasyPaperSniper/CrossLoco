
import math
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit.envs.mdp as mdp

from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.envs import RLTaskEnv, RLTaskEnvCfg
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise


import CrossLoco.envs.mdp as cl_mdp
from CrossLoco.envs.scenes import *


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",  joint_names=[".*"], scale=0.25, use_default_offset=True)



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class HumanMotionCfg(ObsGroup):
        human_motion = ObsTerm(func=cl_mdp.human_input)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    
    
    @configclass
    class SensoryCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel) # duplicate but just to make my comfort
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # observation groups
    human_motion = HumanMotionCfg()
    sensory_obs = SensoryCfg()

    

@configclass
class RandomizationCfg:
    """Configuration for randomization."""
    # startup
    reset_to_ref = RandTerm(func=cl_mdp.reset_to_human_ref)




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    
    # -- cross loco
    cross_loco_r = RewTerm(func=cl_mdp.cycle_consist, weight=1)

    # -- task
    root_pos_tracking = RewTerm(func=cl_mdp.root_pos_track, weight=1)

    
    
    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                             body_names=["Head_lower", "Head_upper", "base", ".*_thigh",".*_calf",".*_hip"]),  
                "threshold": 1.0},
        )
    dof_pos_limits =  RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.5)
  



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                             body_names= ["Head_lower", "Head_upper", "base", ".*_thigh",".*_calf",".*_hip"]), 
                "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    terrain_levels = None




@configclass
class Go2FlatEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: FlatGroundSceneCfg = Go2FlatSceneCfg(num_envs=8, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        
        self.ref_motion = {'motion_dir': '/home/tianyu/Documents/research/orbit/CrossLoco/human_motion/human_motion.pt',
                           'motion_length': 600,
                           'ref_horizon': 5}
        
        # simulation settings
        self.sim.dt = 1/120
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material


        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt



        
     
    

