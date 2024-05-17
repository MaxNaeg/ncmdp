from typing import Optional

import math
import numpy as np

from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium import spaces


VIEWPORT_W = 600
VIEWPORT_H = 400
FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
LEG_DOWN = 18


class LunarLanderWrapper(LunarLander):
    def __init__(self, 
                 adapt_state: bool=False,
                 vel_coeff: float=100.,
                 init_vel_factor: float=1.,
                 opt_min: bool=True,
                 render_mode: Optional[str] = None,
                 continuous: bool = False,
                 gravity: float = -10.0,
                 enable_wind: bool = False,
                 wind_power: float = 15.0,
                 turbulence_power: float = 1.5,):
    
        # to recover the original lunar lander, set 
        # adapt_state = False, vel_coeff = 0, init_vel_factor = 1
        self.min_neg_vel = 0.
        self.initial_neg_velocity = 0.

        super().__init__(render_mode,
                        continuous,
                        gravity,
                        enable_wind,
                        wind_power,
                        turbulence_power,)
        
        
        # add minimum negative velocity to state
        self.adapt_state = adapt_state
        # coefficient for punishing velocity, set to zero for original reward
        self.vel_coeff = vel_coeff
        # factor to reduce initial velocity
        self.init_vel_factor = init_vel_factor
        # whether to optimize for minimum negative velocity or cumulative negative velocity
        self.opt_min = opt_min

        # adapt observation space

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)

        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)


        # for maximum velocity
        if self.adapt_state:
            low = np.append(low, -10.0)
            high = np.append(high, 10.0)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

    def reset(self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None):

        _, dic = super().reset(seed=seed, options=options)

        # Reduces initial velocity by a factor
        self.lander.linearVelocity.x *= self.init_vel_factor
        self.lander.linearVelocity.y *= self.init_vel_factor

        pos = self.lander.position
        vel = self.lander.linearVelocity

        # make sure that initial satte is correct
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8    

        self.min_neg_vel = - np.sqrt(state[2] * state[2] + state[3] * state[3])
        
        # make sure that prev_shaping is correct for new velocity
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        self.prev_shaping = shaping

        if self.adapt_state:
            state.append(0.)


        # for bookkeeping
        self.initial_neg_velocity = self.min_neg_vel

        return state, dic
    

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        neg_vel = - np.sqrt(state[2] * state[2] + state[3] * state[3])

        if self.opt_min:
            rew_adapted = reward + min(self.vel_coeff * (neg_vel - self.min_neg_vel), 0)
        else:
            # square as this reward is integrated over time
            rew_adapted = reward - self.vel_coeff * neg_vel**2

        self.min_neg_vel = min(neg_vel, self.min_neg_vel)

        # adapt state with min negative velocity so far
        if self.adapt_state:
            state = np.append(state, self.min_neg_vel)


        info['min_neg_vel'] = self.min_neg_vel
        info['rew_not_adapted'] = reward
        info['initial_neg_velocity'] = self.initial_neg_velocity


        return state, rew_adapted, terminated, truncated, info

        