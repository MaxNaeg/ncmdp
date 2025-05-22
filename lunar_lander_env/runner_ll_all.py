
import argparse
import datetime
import os
import random
import copy

import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv

from callbacks_ll import EvalCallback_LLWrapper, TensorboardCallbackLL

from lunar_lander_wrapper import LunarLanderWrapper


parser = argparse.ArgumentParser(description="Runner Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = vars(parser.parse_args())


log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


seed=args["seed"]
th.manual_seed(seed)
th.cuda.manual_seed(seed)

random.seed(seed)
np.random.seed(seed)

tensorboard_folder = "./tensorboard/"
best_model_path = "./best_model/"
saved_model_path = "./saved_models/"

def main():

    # PPO paras
    fac_env = 2

    fac_len = 16 * 2
    fac_steps = 4

    n_timesteps= 1e6 * fac_len

    n_envs = 16 * fac_env
    learning_rate = 3e-4
    n_steps = 1024 * fac_steps
    batch_size = 64 * fac_env * fac_steps
    n_epochs= 4 
    gamma = 0.999
    gae_lambda = 0.98
    clip_range = 0.2
    normalize_advantage = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # log paras
    eval_freq = int(1e5 / n_envs)
    verbose = 0

    def linear_annel_lr(progress):
        return learning_rate * progress


    # Policy params
    ortho_init = False
    net_arch = dict(pi=[256, 256], vf=[256, 256])
    activation_fn = th.nn.Tanh


    # Train with max velocity penalty in last reward
    ppo_coeffs = [12.5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    
    for coeff in ppo_coeffs:
        # env paras
        env_kwargs = {"enable_wind": True,
                    "adapt_state": True,
                    "vel_coeff": coeff,
                    "init_vel_factor": 1.,
                    "opt_min": False,
                    "wind_power": 5, #15.0,
                    "turbulence_power": 0.5, #1.5,
                    "last_reward_vel_max": True
                    }
        
        policy_kwargs = dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
        )

        to_add = "ppo_add_obs_laststep"
        tb_log_name = str(env_kwargs)  + f"seed_{seed}" + to_add
    
        vec_env = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
                            env_kwargs=env_kwargs, seed=seed)
        
        vec_env_callback = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
                            env_kwargs=env_kwargs, seed=seed)
        

        eval_callback = EvalCallback_LLWrapper(vec_env_callback, best_model_save_path=best_model_path, 
                                               log_path='./logs/',
                                    eval_freq=eval_freq, deterministic=False, 
                                    render=False, n_eval_episodes=n_envs,
                                    model_save_name=tb_log_name)

        tensorborad_callback = TensorboardCallbackLL()

        model = PPO("MlpPolicy", vec_env, tensorboard_log=tensorboard_folder,
                    learning_rate = copy.deepcopy(linear_annel_lr),
                    n_steps = n_steps,
                    batch_size = batch_size,
                    n_epochs = n_epochs,
                    gamma = gamma,
                    gae_lambda = gae_lambda,
                    clip_range = clip_range,
                    normalize_advantage = normalize_advantage,
                    ent_coef = ent_coef,
                    vf_coef = vf_coef,
                    max_grad_norm = max_grad_norm,
                    policy_kwargs = policy_kwargs,
                    verbose = verbose,
                    seed = seed,)

        model.learn(total_timesteps=int(n_timesteps), progress_bar=False, tb_log_name=tb_log_name,
                    callback=[eval_callback, tensorborad_callback])
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model.save(saved_model_path + tb_log_name + current_time)
    

    #Train MINPPO with observation
    # ppo_coeffs = [25, 50, 100, 200, 400, 800, 1600]

    # ppo_coeffs = [12.5, 75, 150, 300, 600, 1200, 2400]
    # ppo_coeffs = [12.5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    
    # for coeff in ppo_coeffs:
    #     # env paras
    #     env_kwargs = {"enable_wind": True,
    #                 "adapt_state": True,
    #                 "vel_coeff": coeff,
    #                 "init_vel_factor": 1.,
    #                 "opt_min": True,
    #                 "wind_power": 5, #15.0,
    #                 "turbulence_power": 0.5, #1.5,
    #                 }
        
    #     policy_kwargs = dict(
    #             net_arch=net_arch,
    #             activation_fn=activation_fn,
    #             ortho_init=ortho_init,
    #     )

    #     to_add = "maxppo_obs"
    #     tb_log_name = str(env_kwargs)  + f"seed_{seed}" + to_add
    
    #     vec_env = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        
    #     vec_env_callback = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        

    #     eval_callback = EvalCallback_LLWrapper(vec_env_callback, best_model_save_path=best_model_path, 
    #                                            log_path='./logs/',
    #                                 eval_freq=eval_freq, deterministic=False, 
    #                                 render=False, n_eval_episodes=n_envs,
    #                                 model_save_name=tb_log_name)

    #     tensorborad_callback = TensorboardCallbackLL()

    #     model = PPO("MlpPolicy", vec_env, tensorboard_log=tensorboard_folder,
    #                 learning_rate = copy.deepcopy(linear_annel_lr),
    #                 n_steps = n_steps,
    #                 batch_size = batch_size,
    #                 n_epochs = n_epochs,
    #                 gamma = gamma,
    #                 gae_lambda = gae_lambda,
    #                 clip_range = clip_range,
    #                 normalize_advantage = normalize_advantage,
    #                 ent_coef = ent_coef,
    #                 vf_coef = vf_coef,
    #                 max_grad_norm = max_grad_norm,
    #                 policy_kwargs = policy_kwargs,
    #                 verbose = verbose,
    #                 seed = seed,)

    #     model.learn(total_timesteps=int(n_timesteps), progress_bar=False, tb_log_name=tb_log_name,
    #                 callback=[eval_callback, tensorborad_callback])
    #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     model.save(saved_model_path + tb_log_name + current_time)

    # #Train MINPPO without observation
    # ppo_coeffs = [12.5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    # for coeff in ppo_coeffs:
    #     # env paras
    #     env_kwargs = {"enable_wind": True,
    #                 "adapt_state": False,
    #                 "vel_coeff": coeff,
    #                 "init_vel_factor": 1.,
    #                 "opt_min": True,
    #                 "wind_power": 5, #15.0,
    #                 "turbulence_power": 0.5, #1.5,
    #                 }
        
    #     policy_kwargs = dict(
    #             net_arch=net_arch,
    #             activation_fn=activation_fn,
    #             ortho_init=ortho_init,
    #     )


    #     to_add = "maxppo_no_obs"
    #     tb_log_name = str(env_kwargs)  + f"seed_{seed}" + to_add
    
    #     vec_env = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        
    #     vec_env_callback = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        

    #     eval_callback = EvalCallback_LLWrapper(vec_env_callback, best_model_save_path=best_model_path, 
    #                                            log_path='./logs/',
    #                                 eval_freq=eval_freq, deterministic=False, 
    #                                 render=False, n_eval_episodes=n_envs,
    #                                 model_save_name=tb_log_name)

    #     tensorborad_callback = TensorboardCallbackLL()

    #     model = PPO("MlpPolicy", vec_env, tensorboard_log=tensorboard_folder,
    #                 learning_rate = copy.deepcopy(linear_annel_lr),
    #                 n_steps = n_steps,
    #                 batch_size = batch_size,
    #                 n_epochs = n_epochs,
    #                 gamma = gamma,
    #                 gae_lambda = gae_lambda,
    #                 clip_range = clip_range,
    #                 normalize_advantage = normalize_advantage,
    #                 ent_coef = ent_coef,
    #                 vf_coef = vf_coef,
    #                 max_grad_norm = max_grad_norm,
    #                 policy_kwargs = policy_kwargs,
    #                 verbose = verbose,
    #                 seed = seed,)

    #     model.learn(total_timesteps=int(n_timesteps), progress_bar=False, tb_log_name=tb_log_name,
    #                 callback=[eval_callback, tensorborad_callback])
    #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     model.save(saved_model_path + tb_log_name + current_time)

    # # Train PPO with negative velocity penalty
    # ppo_coeffs = [0., 0.125, 0.25, 0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16., 24.,]

    # for coeff in ppo_coeffs:
    #     # env paras
    #     env_kwargs = {"enable_wind": True,
    #                 "adapt_state": False,
    #                 "vel_coeff": coeff,
    #                 "init_vel_factor": 1.,
    #                 "opt_min": False,
    #                 "wind_power": 5, #15.0,
    #                 "turbulence_power": 0.5, #1.5,
    #                 }
        
    #     policy_kwargs = dict(
    #             net_arch=net_arch,
    #             activation_fn=activation_fn,
    #             ortho_init=ortho_init,
    #     )

    #     #env_kwargs = {}

    #     to_add = "ppo_normal"
    #     tb_log_name = str(env_kwargs)  + f"seed_{seed}" + to_add
    
    #     vec_env = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        
    #     vec_env_callback = make_vec_env(LunarLanderWrapper, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, 
    #                         env_kwargs=env_kwargs, seed=seed)
        

    #     eval_callback = EvalCallback_LLWrapper(vec_env_callback, best_model_save_path=best_model_path, 
    #                                            log_path='./logs/',
    #                                 eval_freq=eval_freq, deterministic=False, 
    #                                 render=False, n_eval_episodes=n_envs,
    #                                 model_save_name=tb_log_name)

    #     tensorborad_callback = TensorboardCallbackLL()

    #     model = PPO("MlpPolicy", vec_env, tensorboard_log=tensorboard_folder,
    #                 learning_rate = copy.deepcopy(linear_annel_lr),
    #                 n_steps = n_steps,
    #                 batch_size = batch_size,
    #                 n_epochs = n_epochs,
    #                 gamma = gamma,
    #                 gae_lambda = gae_lambda,
    #                 clip_range = clip_range,
    #                 normalize_advantage = normalize_advantage,
    #                 ent_coef = ent_coef,
    #                 vf_coef = vf_coef,
    #                 max_grad_norm = max_grad_norm,
    #                 policy_kwargs = policy_kwargs,
    #                 verbose = verbose,
    #                 seed = seed,)

    #     model.learn(total_timesteps=int(n_timesteps), progress_bar=False, tb_log_name=tb_log_name,
    #                 callback=[eval_callback, tensorborad_callback])
    #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     model.save(saved_model_path + tb_log_name + current_time)

if __name__ == '__main__':
    main()