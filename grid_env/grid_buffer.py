import numpy as np
import tensorflow as tf

class Buffer():
    def __init__(self, size, obs_size):
        self.step = 0
        self.size = size
        self.obs = np.zeros((size, obs_size))
        self.new_obs = np.zeros((size, obs_size))
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.actions = np.zeros(size)

    def add(self, obs, obs_new, action, reward, done):
        self.obs[self.step] = obs
        self.new_obs[self.step] = obs_new
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.actions[self.step] = action
        self.step +=1
        self.step = (self.step + 1 ) % self.size
    
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size)
        return (tf.constant(self.obs[idx], dtype=tf.float32), 
                tf.constant(self.new_obs[idx], dtype=tf.float32), 
                tf.constant(self.actions[idx], dtype=np.int32),
                tf.constant(self.rewards[idx], dtype=tf.float32), 
                tf.constant(self.dones[idx], dtype=tf.float32))

