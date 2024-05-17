
import numpy as np

class MaxWrapper:
    def __init__(self, env, adapt_state:bool, adapt_reward:bool) -> None:
        self.env = env
        self.adapt_state = adapt_state
        self.adapt_reward = adapt_reward

    def reset(self):
        self.x = 0
        obs, reward, done = self.env.reset()
        if self.adapt_state:
            obs = np.concatenate((obs, [self.x]))
        return obs, reward, done
    
    def step(self, action):
        obs, reward, done = self.env.step(action)

        if reward + self.x > 0:
            reward_t = reward + self.x
            self.x = 0
        else:
            reward_t = 0
            self.x += reward

        if self.adapt_state:
            obs = np.concatenate((obs, [self.x]))
        
        if self.adapt_reward:
            reward = reward_t

        return obs, reward, done