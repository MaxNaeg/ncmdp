
import torch as th
import numpy as np
import os
import copy
import pickle



from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import SubprocVecEnv

from lunar_lander_wrapper import LunarLanderWrapper

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_performance_ll_wrapper(env, model, n_eval_episodes, cutoff_length):
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_rewards_not_adapted = []
    episode_neg_min_vel = []

    episode_init_vel, episode_sucess = [], []
    n_cuttoff = 0

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_rewards_not_adapted = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    # maximal steps for eval
    max_total_steps = cutoff_length
    total_steps = 0
    print("start evaluation...", flush=True)
    while (episode_counts < episode_count_targets).any():
        total_steps+=1
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=False,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_rewards_not_adapted += np.array([infos[i]["rew_not_adapted"] for i in range(n_envs)])
        current_lengths += 1
        
        for i in range(n_envs):
            if current_lengths[i] > max_total_steps:
                # reset subenv
                env.remotes[i].send(("reset", (env._seeds[i], env._options[i])))
                obs_i, state_i = env.remotes[i].recv()
                #obs_i, state_i = env.reset(i)
                n_cuttoff+=1
                current_rewards[i] = 0
                current_lengths[i] = 0
                current_rewards_not_adapted[i] = 0
                new_observations[i] = obs_i
                #states[i] = state_i

            elif episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done


                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                    episode_rewards_not_adapted.append(current_rewards_not_adapted[i]) 
                    episode_neg_min_vel.append(info["min_neg_vel"])
                    episode_init_vel.append(info["initial_neg_velocity"])
                    episode_sucess.append(True if current_rewards_not_adapted[i] > 0 else False)

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_rewards_not_adapted[i] = 0

        observations = new_observations

    print("finish evaluation...", flush=True)

    return (episode_rewards, episode_lengths, episode_rewards_not_adapted, episode_neg_min_vel, 
            episode_init_vel, episode_sucess, n_cuttoff)






def main():

    (episode_rewards_all, episode_lengths_all, episode_rewards_not_adapted_all, episode_neg_min_vel_all, 
    episode_init_vel_all, episode_sucess_all, n_cuttoff_all) = [], [], [], [], [], [], []
    seeds = np.arange(5)
    save_path = "saved_models/"
    n_eval_episodes = 10000
    cutoff_length=1e5

    for seed_np in seeds:
        episode_rewards_all.append([])
        episode_lengths_all.append([])
        episode_rewards_not_adapted_all.append([])
        episode_neg_min_vel_all.append([])
        episode_init_vel_all.append([])
        episode_sucess_all.append([])
        n_cuttoff_all.append([])

        print(f"seed: {seed_np}", flush=True)

        seed = int(seed_np)

        fac_env = 2

        fac_len = 16
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
            return learning_rate * (1 - progress)


        # Policy params
        ortho_init = False
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        activation_fn = th.nn.Tanh
        

        # # # change starting here
        to_add = "maxppo_no_obs"
        ppo_coeffs = [12.5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    
        for coeff in ppo_coeffs:
            # env paras
            env_kwargs = {"enable_wind": True,
                    "adapt_state": False,
                    "vel_coeff": coeff,
                    "init_vel_factor": 1.,
                    "opt_min": True,
                    "wind_power": 5, #15.0,
                    "turbulence_power": 0.5, #1.5,
                    }

        # to_add = "ppo_normal"
        # # ppo_coeffs = [0., 0.25, 0.5, 1., 2., 4., 8., 16.]
        # # ppo_coeffs = [0.125, 0.75, 1.5, 3., 6., 12., 24.] 
        # ppo_coeffs = [0., 0.125, 0.25, 0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16., 24.,]
    
        # for coeff in ppo_coeffs:
        #     # env paras
        #     env_kwargs = {"enable_wind": True,
        #             "adapt_state": False,
        #             "vel_coeff": coeff,
        #             "init_vel_factor": 1.,
        #             "opt_min": False,
        #             "wind_power": 5, #15.0,
        #             "turbulence_power": 0.5, #1.5,
        #             }

        # to_add = "maxppo_obs"
        # # ppo_coeffs = [25, 50, 100, 200, 400, 800, 1600]
        # # ppo_coeffs = [12.5, 75, 150, 300, 600, 1200, 2400]
        # ppo_coeffs = [12.5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400]
            
        # for coeff in ppo_coeffs:
        #     # env paras
        #     env_kwargs = {"enable_wind": True,
        #             "adapt_state": True,
        #             "vel_coeff": coeff,
        #             "init_vel_factor": 1.,
        #             "opt_min": True,
        #             "wind_power": 5, #15.0,
        #             "turbulence_power": 0.5, #1.5,
        #             }
        
        
            # stop changing here
            tb_log_name = str(env_kwargs)  + f"seed_{seed}" + to_add

            # TODO: check this list if our path is good
            path_l = [save_path + model for model in os.listdir(save_path) if tb_log_name in model]

            if len(path_l) >= 1:
                rel_path = path_l[0]
            
                policy_kwargs = dict(
                        net_arch=net_arch,
                        activation_fn=activation_fn,
                        ortho_init=ortho_init,
                )

            
                vec_env = make_vec_env(LunarLanderWrapper, n_envs=n_envs, vec_env_cls=SubprocVecEnv, 
                                    env_kwargs=env_kwargs, seed=seed)


                model = PPO("MlpPolicy", vec_env, tensorboard_log="./tensorboard_big_exp_2/",
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
                model.set_parameters(rel_path)

                (episode_rewards, episode_lengths, episode_rewards_not_adapted, episode_neg_min_vel, 
                episode_init_vel, episode_sucess, n_cuttoff) = evaluate_performance_ll_wrapper(vec_env, model, n_eval_episodes=n_eval_episodes, cutoff_length=cutoff_length)

                episode_rewards_all[seed].append(episode_rewards)
                episode_lengths_all[seed].append(episode_lengths)
                episode_rewards_not_adapted_all[seed].append(episode_rewards_not_adapted)
                episode_neg_min_vel_all[seed].append(episode_neg_min_vel)
                episode_init_vel_all[seed].append(episode_init_vel)
                episode_sucess_all[seed].append(episode_sucess)
                n_cuttoff_all[seed].append(n_cuttoff)
            else: 
                print(f"Model not found for {tb_log_name}!")


    save_path= "eval_results/"
    with open(save_path + f"episode_rewards_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_rewards_all, f)

    with open(save_path + f"episode_lengths_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_lengths_all, f)

    with open(save_path + f"episode_rewards_not_adapted_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_rewards_not_adapted_all, f)

    with open(save_path + f"episode_neg_min_vel_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_neg_min_vel_all, f)

    with open(save_path + f"episode_init_vel_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_init_vel_all, f)

    with open(save_path + f"episode_sucess_all_{to_add}.pkl", "wb") as f:
        pickle.dump(episode_sucess_all, f)

    with open(save_path + f"n_cuttoff_all_{to_add}.pkl", "wb") as f:
        pickle.dump(n_cuttoff_all, f)




if __name__ == '__main__':
    main()
