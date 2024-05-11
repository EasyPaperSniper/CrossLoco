from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit.envs.mdp as mdp
from dataclasses import MISSING
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.envs import RLTaskEnv, RLTaskEnvCfg


import transmimicV2_interaction.envs.mdp as tm_mdp
from transmimicV2_interaction.envs.scenes import *



HUMAN_INFO = ['human', 1, 1,SMPL_KEY_LINK_IDX] # [character_name, height_norm_vec]
CHILD_INFO = ['child', 0.5, 0.5,SMPL_KEY_LINK_IDX] # [character_name, height_norm_vec]
Go2Ar_INFO = ['Go2Ar', 0.8, 0.3,GO2AR_KEY_LINK_IDX] # [character_name, height_norm_vec]




####################### Action ########################
@configclass
class SingleCharActionsCfg:
    """Action specifications for the environment."""
    c1_action = mdp.RelativeJointPositionActionCfg(asset_name="c1", joint_names=[".*"], scale=0.25)



@configclass
class DoubleCharActionsCfg(SingleCharActionsCfg):
    """Action specifications for the environment."""
    c2_action = mdp.RelativeJointPositionActionCfg(asset_name="c2", joint_names=[".*"], scale=0.25)
    


####################### Observation ########################
@configclass
class CharSensoryInfoC1(ObsGroup):
    """Observations for policy group."""
    
    projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("c1")})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("c1")})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("c1")})
    c1_actions = ObsTerm(func=mdp.last_action, params={'action_name':'c1_action'})

    
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CharSensoryInfoC2(ObsGroup):   # very ugly, try to find a way to make it look better
    """Observations for policy group."""
    projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("c2")})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("c2")})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("c2")})
    c2_actions = ObsTerm(func=mdp.last_action, params={'action_name':'c2_action'})

    
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class WorldObserver(ObsGroup):
    
    c1_ref = ObsTerm(func=tm_mdp.get_ref_motion_w, params={'asset_name': 'c1'})
    c1_pos_w = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("c1")})
    c1_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("c1")})
    c1_base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("c1")})
    c1_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("c1")})
    c1_root_height = ObsTerm(func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("c1")})
    

    c2_ref = ObsTerm(func=tm_mdp.get_ref_motion_w, params={'asset_name': 'c2'})
    c2_pos_w = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("c2")})
    c2_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("c2")})
    c2_base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("c2")})
    c2_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("c2")})
    c2_root_height = ObsTerm(func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("c2")})
    
    
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        
        
    
@configclass
class SingleCharObservationsCfg:
    """Observation specifications for the environment."""
    c1_obs: CharSensoryInfoC1 = CharSensoryInfoC1()


@configclass
class DoubleCharObservationsCfg(SingleCharObservationsCfg):
    c2_obs: CharSensoryInfoC2 = CharSensoryInfoC2()
    world_obs: WorldObserver = WorldObserver()


####################### Reward ########################
@configclass
class SingleCharRewardsCfg:
    # tracking reward
    c1_tracking = RewTerm(func=tm_mdp.tracking_reward, weight=1., params={'asset_name': 'c1'})
    c1_energy_penlty = RewTerm(func=tm_mdp.energy_penlty, weight=-0.001, params={'asset_name': 'c1'})
    c1_ori_tracking = RewTerm(func=tm_mdp.root_ori_tracking, weight=.25, params={'asset_name': 'c1'})
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
   
    
    



@configclass
class DoubleCharRewardsCfg(SingleCharRewardsCfg):
    c2_tracking = RewTerm(func=tm_mdp.tracking_reward, weight=1., params={'asset_name': 'c2'})
    c2_energy_penlty = RewTerm(func=tm_mdp.energy_penlty, weight=-0.001, params={'asset_name': 'c2'})
    c2_ori_tracking = RewTerm(func=tm_mdp.root_ori_tracking, weight=.25, params={'asset_name': 'c2'})
    # AIG_tracking = RewTerm(func=tm_mdp.AIG_reward, weight=3)
    # crs_reward = RewTerm(func=tm_mdp.Xmorph_reward, weight=0.1)


    

####################### Randomization #################################
@configclass
class SingleCharRandomizationCfg:
    """Configuration for randomization."""
    reset_scene = RandTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_motion = RandTerm(func=tm_mdp.sample_reference, mode="reset")
    reset_to_motion_c1 = RandTerm(func=tm_mdp.reset_to_motion_init, params={'asset_name': 'c1'}, mode="reset")


@configclass
class DoubleCharRandomizationCfg(SingleCharRandomizationCfg):
    """Configuration for randomization."""
    reset_to_motion_c2 = RandTerm(func=tm_mdp.reset_to_motion_init, params={'asset_name': 'c2'}, mode="reset")
    


####################### Termination #################################
@configclass
class SingleCharTerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Out of track
    c1_joint_pos_w_off_track = DoneTerm(func=tm_mdp.joint_pos_w_off_track,params={'asset_name': 'c1', "threshold": .5})
    c1_joint_pos_w_off_track_max = DoneTerm(func=tm_mdp.joint_pos_w_off_track_max,params={'asset_name': 'c1', "threshold": 1.})
    c1_root_ori_off_max = DoneTerm(func=tm_mdp.root_ori_w_off_max, params={'asset_name': 'c1', "threshold": 1.})
    # c1_base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_c1", 
    #                                          body_names=["Pelvis",
    #                                                      "link0_4.*","link0_5.*",
    #                                                      "link0_6.*","link0_7.*","link0_8.*"]), 
    #                                         "threshold": 25.0},
    # )



@configclass
class DoubleCharTerminationsCfg(SingleCharTerminationsCfg):
    """Termination terms for the MDP."""
    # (2) Out of track
    c2_joint_pos_w_off_track = DoneTerm(func=tm_mdp.joint_pos_w_off_track,params={'asset_name': 'c2', "threshold": .5})
    c2_joint_pos_w_off_track_max = DoneTerm(func=tm_mdp.joint_pos_w_off_track_max,params={'asset_name': 'c2', "threshold": 1.})
    c2_root_ori_off_max = DoneTerm(func=tm_mdp.root_ori_w_off_max, params={'asset_name': 'c2', "threshold": 1.})
    # aig_off_max = DoneTerm(func=tm_mdp.AIG_off_max, params={"threshold": .3})
    # aig_center_off = DoneTerm(func=tm_mdp.AIG_center_off, params={"threshold": .5})




####################### Command #################################
@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # no commands for this MDP
    null = mdp.NullCommandCfg()


####################### Curriculum #################################
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass


@configclass
class HumanXHumanEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: HumanXHumanSceneCfg = HumanXHumanSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)
    # Basic settings
    observations: DoubleCharObservationsCfg = DoubleCharObservationsCfg()
    actions: DoubleCharActionsCfg = DoubleCharActionsCfg()
    randomization: DoubleCharRandomizationCfg = DoubleCharRandomizationCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: DoubleCharRewardsCfg = DoubleCharRewardsCfg()
    terminations: DoubleCharTerminationsCfg = DoubleCharTerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()
    
    char_property = {'c1':HUMAN_INFO,   
                     'c2':HUMAN_INFO} 
    
    ref_motion = {
        'motion_name' : 'sparring',
        'motion_length': 200,
        'ref_horizon':4
    }

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (2.0, -2.0, 2.0)
        # simulation settings
        self.sim.dt = 1 / 120


@configclass
class ChildXChildEnvCfg(HumanXHumanEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: ChildXChildSceneCfg = ChildXChildSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)
    char_property = {'c1':CHILD_INFO, 
                     'c2':CHILD_INFO} 
    
    
    
@configclass
class Go2ArXGo2ArEnvCfg(HumanXHumanEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene = Go2ArXGo2ArSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)
    char_property = {'c1':Go2Ar_INFO, 
                     'c2':Go2Ar_INFO} 
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        self.actions.c1_action = mdp.JointPositionActionCfg(asset_name="c1", joint_names=[".*"], scale=0.25, use_default_offset=True)
        self.actions.c2_action = mdp.JointPositionActionCfg(asset_name="c2", joint_names=[".*"], scale=0.25, use_default_offset=True)


        self.rewards.c1_tracking = None
        self.rewards.c2_tracking = None
        self.rewards.c1_energy_penlty = None
        self.rewards.c2_energy_penlty = None


        self.rewards.c1_ori_tracking.weight = 0.1
        self.rewards.c2_ori_tracking.weight = 0.1
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

        self.rewards.c1_height_tracking = RewTerm(func=tm_mdp.root_height_tracking, weight=.1, params={'asset_name': 'c1','tgt_height':0.3})
        self.rewards.c2_height_tracking = RewTerm(func=tm_mdp.root_height_tracking, weight=.1, params={'asset_name': 'c2','tgt_height':0.3})
        self.rewards.c1_flat = RewTerm(func=tm_mdp.flat_orientation_l2, weight=-.5, params={'asset_name': 'c1','std':1})
        self.rewards.c2_flat = RewTerm(func=tm_mdp.flat_orientation_l2, weight=-.5, params={'asset_name': 'c2','std':1})
        self.rewards.c1_hip_pos = RewTerm(
            func=tm_mdp.track_joint_pos_target_l2,
            weight= -0.1, # 0.25
            params={"asset_cfg": SceneEntityCfg("c1", joint_names=[".*hip.*"]), "target": 0.0},
            )
        self.rewards.c2_hip_pos = RewTerm(
            func=tm_mdp.track_joint_pos_target_l2,
            weight= -0.1, # 0.25
            params={"asset_cfg": SceneEntityCfg("c2", joint_names=[".*hip.*"]), "target": 0.0},
            )
        
        # self.rewards.c1_arm_base_pos = RewTerm(
        #     func=tm_mdp.track_joint_pos_target_l2,
        #     weight= -0.115,
        #     params={"asset_cfg": SceneEntityCfg("c1", joint_names=["joint2"]), "target": 2.8},
        #     )
        # self.rewards.c2_arm_base_pos = RewTerm(
        #     func=tm_mdp.track_joint_pos_target_l2,
        #     weight= -0.115, 
        #     params={"asset_cfg": SceneEntityCfg("c2", joint_names=["joint2"]), "target": 2.8},
        #     )
        
        self.rewards.c1_feet_air_time = RewTerm(
            func=tm_mdp.feet_air_time,
            weight= 0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces_c1", body_names= ".*_foot"),
                "threshold": 0.75, }, )
        self.rewards.c2_feet_air_time = RewTerm(
            func=tm_mdp.feet_air_time,
            weight= 0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces_c2", body_names= ".*_foot"),
                "threshold": 0.75, }, )
        
        
        self.terminations.c1_root_ori_off_max = None
        self.terminations.c2_root_ori_off_max = None
        self.terminations.c1_root_z_off = DoneTerm(func=tm_mdp.root_off_track_z, params={'asset_name': 'c1', "threshold": .2})
        self.terminations.c2_root_z_off = DoneTerm(func=tm_mdp.root_off_track_z, params={'asset_name': 'c2', "threshold": .2})
        
        self.terminations.c1_base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces_c1", 
                                                body_names= ["Head_lower", "Head_upper", "base", ".*_thigh",".*_calf",".*_hip"]), 
                    "threshold": 1.0},
        )
        self.terminations.c2_base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces_c2", 
                                                body_names= ["Head_lower", "Head_upper", "base", ".*_thigh",".*_calf",".*_hip"]), 
                    "threshold": 1.0},
        )
