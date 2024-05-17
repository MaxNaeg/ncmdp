import numpy as np

class PillarEnv:
    def __init__(self, size:int, top:float, diff:float, max_steps:int, rng:np.random.default_rng, 
                 teleport:bool=False, wind_prob:float=0., one_hot=False) -> None:
        assert size%2 == 0, "size must be even number" 
        self.size = size
        self.top = top
        self.diff = diff
        self.max_steps = max_steps
        self.rng = rng
        self.teleport = teleport
        self.wind_prob = wind_prob
        self.one_hot = one_hot
        
        self.cost_list = np.zeros(self.size * 2 + 1, dtype=np.float32)
        self.cost_list[0:self.size+1:2] = np.linspace(self.top, 0, self.size//2+1, dtype=np.float32)
        self.cost_list[1:self.size:2] = np.linspace(self.top + self.diff, self.diff, self.size//2, dtype=np.float32)
        self.cost_list[self.size::2] = np.linspace(0, self.top, self.size//2+1, dtype=np.float32)
        self.cost_list[self.size+1::2] = np.linspace(diff, self.top + self.diff, self.size//2, dtype=np.float32)

        _ = self.reset()
        
    def reset(self) -> float:
        self.step_count = 0
        self.position = self.rng.integers(0, 2 * self.size + 1)
        self.done = 0
        self.last_cost = self.cost_list[self.position]
        self.current_cost = self.last_cost
        # Observation, Reward, Done
        return self.get_observation() , 0., self.done
    
    def get_observation(self) -> int:
        if self.one_hot:
            return np.eye(2*self.size+1)[self.position]
        else:
            return self.position-self.size
    
    def step(self, action):

        if self.step_count >= self.max_steps:
            observation, reward, _ = self.reset()
            #print("bla")
            return observation, reward, 1
        # to make trajectories exactly the same only use rng iof wind is really needed
        if self.wind_prob > 1e-6:
            if self.position%2 and self.rng.random() < self.wind_prob:
                if self.position < self.size:
                    self.position = 0
                else:
                    self.position = 2 * self.size
        
        if self.teleport and self.position == self.size:
            self.position = 0
        elif action == 0:
            self.position = np.max([0, self.position-1])
        elif action == 1:
            self.position = self.position
        elif action == 2:
            self.position = np.min([2 * self.size, self.position+1])
        else:
            raise ValueError("action must be 0, 1, 2")
        self.last_cost = self.current_cost
        self.current_cost = self.cost_list[self.position]

        self.step_count += 1
        # Observation, Reward, Done
        return self.get_observation(), self.last_cost - self.current_cost, self.done
    