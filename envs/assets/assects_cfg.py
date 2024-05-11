import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ActuatorNetMLPCfg, DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

current_dir = os.path.dirname(os.path.realpath(__file__))


ALIENGOZ1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=current_dir+"/aliengoZ1/usd/aliengoZ1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "arm_joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", "arm_joint.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


ALIENGOARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=current_dir+"/aliengoArx5/usd/aliengoArx5.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", "joint.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


SMPLHUMAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=current_dir+"/humanoid/smpl_humanoid/usd/smpl_humanoid.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.95),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=50.,
            velocity_limit=30.0,
            stiffness= {
                '.*_Hip_.*':800,
                '.*_Knee_.*':800,
                '.*_Ankle_.*':800,
                '.*_Toe_.*':500,
                'Torso_.*':1000,
                'Spine_.*':1000,
                'Chest_.*':1000,
                'Neck_.*':500,
                'Head_.*':500,
                '.*_Thorax_.*':500,
                '.*_Shoulder_.*':500,
                '.*_Elbow_.*':500,
                '.*_Wrist_.*':300,
            },
            
            damping={
                '.*_Hip_.*':80,
                '.*_Knee_.*':80,
                '.*_Ankle_.*':80,
                '.*_Toe_.*':50,
                'Torso_.*':100.0,
                'Spine_.*':100.0,
                'Chest_.*':100.0,
                'Neck_.*':50,
                'Head_.*':50,
                '.*_Thorax_.*':50,
                '.*_Shoulder_.*':50,
                '.*_Elbow_.*':50,
                '.*_Wrist_.*':30,
            },
            friction=0.0,
        ),

    },
)


CHILD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=current_dir+"/humanoid/child/usd/child.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.95),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)



GO2ARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{current_dir}/Go2Arx5/usd_fixed_idx/Go2Arx5.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=25.,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm": DCMotorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

# robot = ArticulationCfg(
