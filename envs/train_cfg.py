import inspect

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)




@configclass
class TM2PpoActorCriticCfg():
    
    init_noise_std = 1.
    actor_latent_dim = 64
    actor_encoder_hidden_dims = [1024, 1024, 512]
    actor_decoder_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [1024, 1024, 512]
    activation="lrelu"
    share_primitive=True



@configclass  
class TM2PpoAlgorithmCfg():

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-4 #5.e-4
    schedule = 'adaptive' # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.


@configclass  
class TM2XmorphCfg():
    mapper_dims = [512, 128, 512]
    activation = 'lrelu'
    num_learning_epochs = 2
    num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-4 #5.e-4


        
@configclass
class TM2PPORunnerCfg():    
    seed: int = 7777
    device: str = "cuda:0"
    num_steps_per_env: int = 25
    max_iterations: int = 10000
    empirical_normalization: bool = False
    save_interval = 2000
    experiment_name: str = "Sparring"
    run_name: str = ""
    logger = None
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    wandb_project = "TM2"
    
    policy_class_name = 'TM2_ActorCritic'
    policy= TM2PpoActorCriticCfg()
    """The policy configuration."""
    
    algorithm_class_name = 'TM2_PPO'
    algorithm = TM2PpoAlgorithmCfg()
    """The algorithm configuration."""
    
    
    Xmorph = TM2XmorphCfg()



