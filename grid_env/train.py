import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from grid_env import GridEnv
from grid_buffer import Buffer

from grid_env_utils import (
    get_reward_fn_constant, 
    get_reward_fn_random,
    eval,
    get_q_vals,
    train_step,
    train_step_min,
)


def main():
    typ = 'ncmdp'

    print(f'{typ=}')

    master_seed = 0

    n_seeds = 10
    runs_per_grid = 5

    assert typ in ['min', 'ncmdp'], 'invalid type'
    # env params
    N = 5

    # network params
    hidden_dim = 128
    n_layers = 2
    learning_rate = 1e-2 

    # training params
    total_steps = 100000
    eps = 0.1 # random action probability
    buffer_size = 1000
    batch_size = 1000
    warmup_steps = batch_size
    assert warmup_steps >= buffer_size # otherwise buffer will not be full
    train_every_steps = 100
    eval_every = train_every_steps

    def get_lr_linear(step, lr_min=1e-7):
        return (total_steps-step) / total_steps * learning_rate + lr_min

    n_eval_tries = 1

    eval_rews_all = []
    final_rews_all = []
    losses_all = []
    cum_rews_all = []

    master_rng = np.random.default_rng(master_seed)

    for seed in range(n_seeds):
        eval_rews_run = []
        final_rews_run = []
        losses_run = []
        cum_rews_run = []

        env_seed = int(seed)
        

        for run in range(runs_per_grid):
            rng = np.random.default_rng(env_seed)

            print(f'seed: {seed}, run: {run}')
            # Completely determines env dynamics
            seed_run = master_rng.integers(0, 2**32-1)
            

            # Used for network initializer
            general_seed = int(seed_run)
            tf.random.set_seed(general_seed)
            np.random.seed(general_seed)
            random.seed(general_seed)


            # build env
            if typ == 'min':
                adapt_reward = False
                adapt_state = False
            elif typ == 'ncmdp':
                adapt_reward = True
                adapt_state = True

            get_reward_fn = get_reward_fn_constant

            grid_env = GridEnv(N, get_reward_fn, rng, adapt_reward, adapt_state)

            # build q network
            if adapt_state:
                input_shape = (3,)
            else:
                input_shape = (2,)
                
            Q_net = keras.Sequential()
            Q_net.add(keras.Input(shape=input_shape))
            for _ in range(n_layers):
                Q_net.add(keras.layers.Dense(hidden_dim, activation='tanh'))
            Q_net.add(keras.layers.Dense(3, activation='linear'))

            optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)

            # compile functions
            if typ == 'min':
                train = tf.function(train_step_min)
            else:
                train = tf.function(train_step)

            get_q_vals_comp = tf.function(get_q_vals)


            # train
            loss_list = [] # training loss
            cum_rews = [] # cumulative rewards
            traj_rew = 0. # reward of current trajectory
            eval_rew = [] # evaluation rewards

            obs = tf.constant([grid_env.reset()], dtype=tf.float32)
            buffer = Buffer(buffer_size, obs.shape[1])

            for i in range(total_steps):

                if rng.uniform() < eps:
                    action = rng.integers(3)
                else:
                    action = np.argmax(get_q_vals_comp(Q_net, obs))
                obs_new, reward, done = grid_env.step(action)
                buffer.add(obs, obs_new, action, reward, done)
                obs = tf.constant([obs_new], dtype=tf.float32)

                #log rewards
                traj_rew += reward
                if done:
                    cum_rews.append(traj_rew)
                    traj_rew = 0
                
                # train
                if i > warmup_steps and i % train_every_steps == 0:
                    obs_batch, obs_new_batch, actions_batch, reward_batch, done_batch = buffer.sample(batch_size)
                    optimizer.lr.assign = get_lr_linear(i)
                    loss = train(Q_net, obs_batch, obs_new_batch, actions_batch, reward_batch, done_batch, optimizer)
                    loss_list.append(loss.numpy())

                # eval with deterministic policy
                if i % eval_every == 0:
                    if adapt_reward:
                        performance =  1. + np.mean([np.sum(eval(Q_net, grid_env, get_q_vals_comp)[2]) for _ in range(n_eval_tries)])
                    else:
                        performance = np.mean([np.min(eval(Q_net, grid_env, get_q_vals_comp)[2]) for _ in range(n_eval_tries)])
                    eval_rew.append(performance)

            if adapt_reward:
                final_performance = 1. + np.mean([np.sum(eval(Q_net, grid_env, get_q_vals_comp)[2]) for _ in range(10000)])
            else:
                final_performance = np.mean([np.min(eval(Q_net, grid_env, get_q_vals_comp)[2]) for _ in range(10000)])

            eval_rews_run.append(eval_rew)
            final_rews_run.append(final_performance)
            losses_run.append(loss_list)
            cum_rews_run.append(cum_rews)

        eval_rews_all.append(eval_rews_run)
        final_rews_all.append(final_rews_run)
        losses_all.append(losses_run)
        cum_rews_all.append(cum_rews_run)

    # save
    np.save(f'data_{N}/eval_rews_{typ}_True.npy', np.array(eval_rews_all, dtype=np.float32))
    np.save(f'data_{N}/final_rews_{typ}_True.npy', np.array(final_rews_all, dtype=np.float32))
    np.save(f'data_{N}/losses_{typ}_True.npy', np.array(losses_all, dtype=np.float32))
    np.save(f'data_{N}/cum_rews_{typ}_True.npy', np.array(cum_rews_all, dtype=np.float32))

        
if __name__ == '__main__': 
    main()