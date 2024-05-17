import numpy as np
import tensorflow as tf

class Buffer():
    """Simple trajectory buffer that stores the trajectory in numpy arrays,
    observations can only be 1D and actions integers"""
    def __init__(self, n_steps:int, obs_size:int=1)->None:
        self.n_steps = n_steps
        self.obs_size = obs_size
        self.reset()
    def reset(self)->None:
        """Initializes buffer"""
        self.actions = np.zeros(dtype=np.int32, shape=(self.n_steps,)) 
        self.rewards = np.zeros(dtype=np.float32, shape=(self.n_steps,))
        self.observation = np.zeros(dtype=np.float32, shape=(self.n_steps, self.obs_size))

    def add_step(self, observation:np.ndarray, action:int, reward:float, step:int)->None:
        """Adds a step to the buffer"""
        self.actions[step]=action
        self.observation[step]=observation
        self.rewards[step]=reward
    def get_trajectory(self)->tuple:
        """Returns the trajectory as a tuple of jaxnumpy arrays"""
        return (tf.constant(self.observation, dtype=tf.float32), 
                tf.constant(self.actions, dtype=tf.int32), 
                tf.constant(self.rewards, dtype=tf.float32))
    
class Buffer_Critic():
    """Simple trajectory buffer that stores the trajectory in numpy arrays,
    observations can only be 1D and actions integers"""
    def __init__(self, n_steps:int, obs_size:int=1)->None:
        self.n_steps = n_steps
        self.obs_size = obs_size
        self.reset()
    def reset(self)->None:
        """Initializes buffer"""
        self.actions = np.zeros(dtype=np.int32, shape=(self.n_steps,)) 
        self.rewards = np.zeros(dtype=np.float32, shape=(self.n_steps,))
        self.observation = np.zeros(dtype=np.float32, shape=(self.n_steps, self.obs_size))
        self.values = np.zeros(dtype=np.float32, shape=(self.n_steps+1,))

    def add_step(self, observation:np.ndarray, action:int, reward:float, value:float, step:int)->None:
        """Adds a step to the buffer"""
        self.actions[step]=action
        self.observation[step]=observation
        self.rewards[step]=reward
        self.values[step]=value
    def add_last_value(self, value:float)->None:
        """Adds the last value to the buffer"""
        self.values[-1]=value

    def get_trajectory(self)->tuple:
        """Returns the trajectory as a tuple of tensorflow constants"""
        return (tf.constant(self.observation, dtype=tf.float32), 
                tf.constant(self.actions, dtype=tf.int32), 
                tf.constant(self.rewards, dtype=tf.float32),
                tf.constant(self.values, dtype=tf.float32))