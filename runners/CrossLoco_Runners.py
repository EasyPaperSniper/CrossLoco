import time
import os
from collections import deque
import statistics


import wandb
import torch
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from CrossLoco.runners.learning_modules import Xmorph_network
from CrossLoco.runners.algorithms import XmorphLearning




class CrossLoco_Runner(OnPolicyRunner):
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        # super().__init__(env, train_cfg, log_dir, device)
        self.env = env
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.Xmorph_cfg = train_cfg["Xmorph"]
        self.device = device

        
        num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.alg_cfg.num_steps_per_env
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        
        h_pose, r_pose = self.env.get_X_pose()
        h_pose_dim, r_pose_dim = h_pose.shape[-1], r_pose.shape[-1]
        Xmorph_net = Xmorph_network(h_pose_dim, r_pose_dim, **self.Xmorph_cfg)
        self.Xmorph_alg = XmorphLearning(Xmorph_net, device=self.device, **self.Xmorph_cfg)
        self.Xmorph_alg.init_storage(self.env.num_envs, self.num_steps_per_env, [h_pose_dim], [r_pose_dim])
        
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.log_dir is not None and self.writer is None and self.cfg["logger"]=='wandb':
            self.writer = wandb.init(project=self.cfg['wandb_project'], group=self.cfg['experiment_name'],name=self.cfg['run_name'], dir=self.log_dir)
        
        
        obs, _ = self.env.reset()
        self.env.episode_length_buf = torch.zeros_like(self.env.episode_length_buf)
        self.alg.actor_critic.train()


        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # transfer policy part
                    actions = self.alg.act(obs,obs)
                    obs, rew, dones, info = self.env.step(actions)
                    
                    obs, rew, dones = (
                        obs.to(self.device),
                        rew.to(self.device),
                        dones.to(self.device),
                    )
                    src_pose, tgt_pose = self.env.get_X_pose()
                    src_pose, tgt_pose = src_pose.to(self.device), tgt_pose.to(self.device)

                    self.alg.process_env_step(rew, dones, info)
                    self.Xmorph_alg.store(src_pose, tgt_pose)

                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in info:
                            ep_infos.append(info['episode'])
                        elif "log" in info:
                            ep_infos.append(info["log"])
                        cur_reward_sum += rew

                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs)
                
            
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            mean_reconst_loss = self.Xmorph_alg.update()
            self.env.update_mappers(self.Xmorph_alg.Xmorph_net)

            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        report_iter, report_time = {}, {}

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                report_iter['Episode/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        report_iter['Loss/value_function'] = locs['mean_value_loss']
        report_iter['Loss/h_reconstruct'] = locs['h_reconst_loss']
        report_iter['Loss/r_reconstruct'] = locs['r_reconst_loss']
        report_iter['Loss/surrogate'] = locs['mean_surrogate_loss']
        report_iter['Loss/learning_rate'] = self.alg.learning_rate
        report_iter['Policy/mean_noise_std'] = mean_std.item()
        report_iter['Perf/total_fps'] = fps
        report_iter['Perf/collection time'] = locs['collection_time']
        report_iter['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            report_iter['Train/mean_reward']= statistics.mean(locs['rewbuffer'])
            report_iter['Train/mean_episode_length']= statistics.mean(locs['lenbuffer'])
            report_time['Train/mean_reward/time']= statistics.mean(locs['rewbuffer'])
            report_time['Train/mean_episode_length/time']= statistics.mean(locs['lenbuffer'])
        # self.writer.log(report_iter, locs['it'])
        # self.writer.log(report_time, self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'h_reconst_state_dict': self.reconst_alg.h_reconst_net.state_dict(),
            'h_reconst_optimizer_state_dict': self.reconst_alg.h_optimizer.state_dict(),
            'r_reconst_state_dict': self.reconst_alg.r_reconst_net.state_dict(),
            'r_reconst_optimizer_state_dict': self.reconst_alg.r_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True, load_reconst=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_reconst:
            self.reconst_alg.h_reconst_net.load_state_dict(loaded_dict['h_reconst_state_dict'])
            self.reconst_alg.r_reconst_net.load_state_dict(loaded_dict['r_reconst_state_dict'])
            self.reconst_alg.h_optimizer.load_state_dict(loaded_dict['h_reconst_optimizer_state_dict'])
            self.reconst_alg.r_optimizer.load_state_dict(loaded_dict['r_reconst_optimizer_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
           
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
            self.reconst_alg.h_reconst_net.to(device)
            self.reconst_alg.r_reconst_net.to(device)
        return self.alg.actor_critic.act_inference, self.reconst_alg.h_reconst_net, self.reconst_alg.r_reconst_net