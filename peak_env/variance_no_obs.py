import argparse
import random
import copy

import keras
import tensorflow as tf
import numpy as np

from keras import Sequential
from keras.layers import Dense

from pathlib import Path


from pillar_env import PillarEnv
from maxwrapper import MaxWrapper
from REINFORCE_Baseline import calc_cumulative_return, REINFORCE_BASE
from buffer import Buffer



parser = argparse.ArgumentParser(description="Runner Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# We used seeds 0 to 9 for the experiments
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = vars(parser.parse_args())

seed=args["seed"]
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)


def run_traj(env, reinforce_agent, max_steps, size):

    buffer=Buffer(n_steps = max_steps, obs_size=2*size+1)
    buffer.reset()
    obs, _, _ = env.reset()
    initial_cost=env.env.cost_list[np.argmax(obs)]

    for step in range(max_steps):
        action = reinforce_agent.sample_action(tf.constant([obs]))
        action = action[0][0].numpy()

        new_obs, reward, _ = env.step(action)
        buffer.add_step(obs, action, reward, step)

        obs=new_obs

    min_cost_found = np.min(env.env.cost_list[np.argmax(buffer.observation, axis=1)])
    min_diff = np.min(np.abs(np.argmax(buffer.observation, axis=1)-env.env.size))

    return buffer, initial_cost, min_cost_found, min_diff


def run_traj_both(env, reinforce_agent, max_steps, size):

    buffer=Buffer(n_steps = max_steps, obs_size=2*size+1)
    buffer_normal=Buffer(n_steps = max_steps, obs_size=2*size+1)
    buffer.reset()
    obs, _, _ = env.reset()
    initial_cost=env.env.cost_list[np.argmax(obs)]

    for step in range(max_steps):
        action = reinforce_agent.sample_action(tf.constant([obs]))
        action = action[0][0].numpy()

        env_x_old = copy.copy(env.x)
        new_obs, reward, _ = env.step(action)
        env_x_new = env.x
        reward_normal = reward + (env_x_new - env_x_old)
        buffer.add_step(obs, action, reward, step)
        buffer_normal.add_step(obs, action, reward_normal, step)

        obs=new_obs

    min_cost_found = np.min(env.env.cost_list[np.argmax(buffer.observation, axis=1)])
    min_diff = np.min(np.abs(np.argmax(buffer.observation, axis=1)-env.env.size))

    return buffer, buffer_normal, initial_cost, min_cost_found, min_diff

def calc_gradients_list(reinforce_agent, buffer_list):
    # Caculate gradients shape = (n_traj_var, max_steps, n_params)
    gradients_all = []
    for buffer in buffer_list:
        observations, actions, rewards = buffer.get_trajectory()
        cum_rewards = reinforce_agent.cum_rew_func(rewards, reinforce_agent.gamma, None)
        grads = calc_grads(observations, cum_rewards, actions, reinforce_agent.model)
        gradients_all.append(grads)
    return tf.convert_to_tensor(gradients_all)

def calc_gradients_list_rein(reinforce_agent, buffer_list, ):
    # Caculate gradients shape = (n_traj_var, max_steps, n_params)
    gradients_all = []
    for buffer in buffer_list:
        observations, actions, rewards = buffer.get_trajectory()
        cum_rewards = reinforce_agent.cum_rew_func(rewards, reinforce_agent.gamma, None)
        grads = calc_grads(observations, cum_rewards, actions, reinforce_agent.model)
        gradients_all.append(grads)
    return tf.convert_to_tensor(gradients_all)


@tf.function
def calc_grads(observations, cum_rewards, actions, model):
    with tf.GradientTape() as tape:
        logits = model(observations)
        norms = tf.math.log(tf.reduce_sum(tf.exp(logits), axis=-1))
        policy_loss = -(tf.gather(logits, actions, axis=1, batch_dims=1) - norms) * cum_rewards
        jac = tape.jacobian(policy_loss, model.trainable_variables)
        # TODO: check this
        grads_flat = [tf.concat([tf.reshape(j[i], [-1]) for j in jac], axis=0) for i in range(len(observations))]
    return tf.convert_to_tensor(grads_flat)



# Env paras
size = 10
top = 1
diff = 1
max_steps = 10

# Agent paras
learning_rate = 2**(-10)
n_trajectories = 60000
gamma=1.

# Calc variance paras
calc_each = 500
n_traj_var = 1000

# Setup env
rng = np.random.default_rng(seed)
env_pill = PillarEnv(size, top, diff, max_steps, rng, one_hot=True)
env = MaxWrapper(env_pill, adapt_state=False, adapt_reward=True)

# Setup agent

actor_net = Sequential([keras.Input(shape=(2*env.env.size+1,)), 
                    Dense(3, activation="linear", 
                        kernel_initializer = tf.keras.initializers.Orthogonal(gain=0.01, seed=seed),
                        bias_initializer=tf.keras.initializers.Zeros())])

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)    


reinforce_agent = REINFORCE_BASE(model=actor_net, optimizer=optimizer, critic=None, critic_optimizer=None, 
                    gamma=gamma, cum_rew_func=calc_cumulative_return, train_critic=False)


save_path = Path("variance_no_obs")

save_path_single = save_path / Path("single_points")



# Variance quantities
# Average cost improvement
var_average_cost_improvement = []
#Average dist to minimum at best point during traj
var_average_min_diff = []
# Average gradient
var_average_gradient = []
var_average_gradient_normal = []
# Average variance wrt to average traj gradients
var_average_variance_trajs = []
var_average_variance_trajs_normal = []
# Average variance wrt to all experiences
var_average_variance_exps = []
var_average_variance_exps_normal = []

# Training quantities
train_average_cost_improvement = []
train_average_min_diff = []

for i in range(n_trajectories):

    if i%calc_each==0:
        # All gradients
        var_all_exp_grads = []
        var_all_exp_grads_normal = []
        # All average traj gradients
        var_all_traj_grads = []
        var_all_traj_grads_normal = []

        var_single_cost_improvement = []
        var_single_min_diff = []
        buffer_list = []
        buffer_list_normal = []
        
        # Run and save trajs
        for j in range(n_traj_var):
            buffer, buffer_normal, initial_cost, min_cost_found, min_diff = run_traj_both(env, reinforce_agent, max_steps, size)
            var_single_cost_improvement.append(initial_cost - min_cost_found)
            var_single_min_diff.append(min_diff)
            buffer_list.append(buffer)
            buffer_list_normal.append(buffer_normal)

        var_average_cost_improvement.append(np.mean(var_single_cost_improvement))
        var_average_min_diff.append(np.mean(var_single_min_diff))

        # MAXREINFORCE Gradients
        # Caculate gradients shape = (n_traj_var, max_steps, n_params)
        gradients = calc_gradients_list(reinforce_agent, buffer_list)
        gradients_normal = calc_gradients_list(reinforce_agent, buffer_list_normal)

        # Average gradient
        avg_grad = tf.reduce_mean(gradients, axis=(0,1))
        avg_grad_normal = tf.reduce_mean(gradients_normal, axis=(0,1))

        var_average_gradient.append(avg_grad)
        var_average_gradient_normal.append(avg_grad_normal)

        # Average variance wrt to traj grad
        average_traj_grads = tf.reduce_mean(gradients, axis=1)
        average_traj_grads_normal = tf.reduce_mean(gradients_normal, axis=1)

        var_variance_trajs = tf.reduce_sum(tf.math.reduce_variance(average_traj_grads, axis=0))
        var_variance_trajs_normal = tf.reduce_sum(tf.math.reduce_variance(average_traj_grads_normal, axis=0))

        var_average_variance_trajs.append(var_variance_trajs)
        var_average_variance_trajs_normal.append(var_variance_trajs_normal)

        # Average variance wrt to all experiences
        grads_exp_concat = tf.reshape(gradients, (n_traj_var * max_steps, gradients.shape[-1]))
        grads_exp_concat_normal = tf.reshape(gradients_normal, (n_traj_var * max_steps, gradients_normal.shape[-1]))
        
        var_variance_exps = tf.reduce_sum(tf.math.reduce_variance(grads_exp_concat, axis=0))
        var_variance_exps_normal = tf.reduce_sum(tf.math.reduce_variance(grads_exp_concat_normal, axis=0))

        var_average_variance_exps.append(var_variance_exps)
        var_average_variance_exps_normal.append(var_variance_exps_normal)

        # save intermediate results
        with open(save_path_single / Path(f"all_gradients_seed{seed}_step{i}.npy"), 'wb') as f:
            np.save(f, np.array(gradients))
        with open(save_path_single / Path(f"all_gradients_normal_seed{seed}_step{i}.npy"), 'wb') as f:
            np.save(f, np.array(gradients_normal))

        with open(save_path_single / Path(f"average_traj_grads_seed{seed}_step{i}.npy"), 'wb') as f:
            np.save(f, np.array(average_traj_grads))
        with open(save_path_single / Path(f"average_traj_grads_normal_seed{seed}_step{i}.npy"), 'wb') as f:
            np.save(f, np.array(average_traj_grads_normal))


    # train step
    buffer, initial_cost, min_cost_found, min_diff = run_traj(env, reinforce_agent, max_steps, size)
    reinforce_agent.train_step(*buffer.get_trajectory(), values=None) 

    train_average_cost_improvement.append(initial_cost - min_cost_found)
    train_average_min_diff.append(min_diff)

var_average_cost_improvement = np.array(var_average_cost_improvement)
var_average_min_diff = np.array(var_average_min_diff)

var_average_gradient = np.array(var_average_gradient)
var_average_variance_trajs = np.array(var_average_variance_trajs)
var_average_variance_exps = np.array(var_average_variance_exps)

var_average_gradient_normal = np.array(var_average_gradient_normal)
var_average_variance_trajs_normal = np.array(var_average_variance_trajs_normal)
var_average_variance_exps_normal = np.array(var_average_variance_exps_normal)


train_average_cost_improvement = np.array(train_average_cost_improvement)
train_average_min_diff = np.array(train_average_min_diff)

# Save results
    
with open(save_path / Path(f'var_average_cost_improvement_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_cost_improvement))

with open(save_path / Path(f'var_average_min_diff_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_min_diff))



with open(save_path / Path(f'var_average_gradient_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_gradient))

with open(save_path / Path(f'var_average_variance_trajs_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_variance_trajs))

with open(save_path / Path(f'var_average_variance_exps_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_variance_exps))

with open(save_path / Path(f'var_average_gradient_normal_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_gradient_normal))

with open(save_path / Path(f'var_average_variance_trajs_normal_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_variance_trajs_normal))

with open(save_path / Path(f'var_average_variance_exps_normal_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(var_average_variance_exps_normal))




with open(save_path / Path(f'train_average_cost_improvement_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(train_average_cost_improvement))

with open(save_path / Path(f'train_average_min_diff_seed{seed}.npy'), 'wb') as f:
    np.save(f, np.array(train_average_min_diff))



    


            