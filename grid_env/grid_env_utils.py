import numpy as np
import tensorflow as tf

# Reward function -------------------------------------------------
def get_reward_fn_constant(pos, rng):
    reward = rng.uniform(-1, 1)
    return lambda rngg: reward


# Eval function ---------------------------------------------------
# record one trajectory
def eval(Q_net, grid_env, get_q_vals_comp):
    act_seq = []
    obs_seq = []
    rews = []
    traj_rew = 0.
    done = False
    obs = tf.constant([grid_env.reset()], dtype=tf.float32)
    while not done:
        action = np.argmax(get_q_vals_comp(Q_net, obs))
        obs_new, reward, done = grid_env.step(action)
        rews.append(reward)
        obs_seq.append(obs.numpy())
        act_seq.append(action)
        traj_rew += reward
        obs_new = tf.constant([obs_new], dtype=tf.float32)
        obs = obs_new
    traj_rew = np.sum(rews)
    return act_seq, obs_seq, rews

# Train functions---------------------------------------------------

#@tf.function
def get_q_vals(Q_net, state):
    return Q_net(state)

# normal Q-learning loss function
def loss_func(Q_net, obs, obs_new, actions, reward, done):
    fac = tf.abs(done - tf.constant([1,], dtype=tf.float32))
    target = reward + fac * tf.reduce_max(Q_net(obs_new), axis=-1)
    actions_selection = tf.one_hot(actions, depth=3)
    loss = tf.reduce_mean(tf.square(target - tf.reduce_sum(Q_net(obs)*actions_selection, axis=-1)))
    return loss

# minimum Q-learning loss function
def loss_func_min(Q_net, obs, obs_new, actions, reward, done):
    fac = tf.abs(done - tf.constant([1,], dtype=tf.float32))
    target = tf.reduce_min(tf.stack([reward, fac * tf.reduce_max(Q_net(obs_new), axis=-1)], axis=-1), axis=-1)
    actions_selection = tf.one_hot(actions, depth=3)
    loss = tf.reduce_mean(tf.square(target - tf.reduce_sum(Q_net(obs)*actions_selection, axis=-1)))
    return loss

#@tf.function
def train_step(Q_net, obs, obs_new, actions, reward, done, optimizer):
    with tf.GradientTape() as tape:
        loss = loss_func(Q_net, obs, obs_new, actions, reward, done)
    grads = tape.gradient(loss, Q_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, Q_net.trainable_variables))
    return loss

#@tf.function
def train_step_min(Q_net, obs, obs_new, actions, reward, done, optimizer):
    with tf.GradientTape() as tape:
        loss = loss_func_min(Q_net, obs, obs_new, actions, reward, done)
    grads = tape.gradient(loss, Q_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, Q_net.trainable_variables))
    return loss