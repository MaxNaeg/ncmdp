import numpy as np

class GridEnv():
    def __init__(self, 
                 N: int, # side length
                 get_reward_fn:callable, #function that returns function that gives reward given a position
                 rng, # random number generator
                 adapt_reward=False, #min reward
                 adapt_state=False #add min reward to state
                 ):
        
        self.get_reward_fn = get_reward_fn
        self.N = N
        self.rng = rng

        self.adapt_reward = adapt_reward
        self.adapt_state = adapt_state

        self.reset()

        # get reward function for each grid point
        self.reward_fns = []
        for i in range(N):
            self.reward_fns.append([])
            for j in range(N):
                self.reward_fns[i].append(get_reward_fn((i, j), self.rng))

        self.reward_fns[self.pos[0]][self.pos[1]] = lambda rngg: 1.

    def get_obs(self):
        if self.adapt_state:
            return np.array(self.pos + [self.min_rew], dtype=np.float32)
        else:
            return np.array(self.pos, dtype=np.float32)

    def reset(self):
        self.pos = [0, self.N//2]
        # set reward for starting position to maximum possible reward
        self.min_rew = 1.
        return self.get_obs()
    
    def step(self, action:int):
        # action: 0: left, 1: stay, 2: right
        assert action in [0, 1, 2], 'invalid action'

        # always step forward
        self.pos[0] += 1
        self.pos[1] += (-1 + action)
        #stay in grid
        self.pos[1] = np.clip(self.pos[1], 0, self.N-1)

        reward = self.reward_fns[self.pos[0]][self.pos[1]](self.rng)

        if self.adapt_reward:
            rew_adapted = min(0, reward - self.min_rew)
        else:
            rew_adapted = reward
        
        self.min_rew = min(self.min_rew, reward)

        # end of grid, final state
        if self.pos[0] == self.N-1:
            done = True
            self.reset()
            # rew_adapted = 0.
            # self.reset()
            # return self.get_obs(), rew_adapted, done
        else:
            done = False
            
        return self.get_obs(), rew_adapted, done

