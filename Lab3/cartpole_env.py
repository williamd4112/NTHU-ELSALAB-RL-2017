import gym
import numpy as np

class CartpoleEnvironment(object):

    MAX_TIMESTEP = 200

    def __init__(self, viz=False):
        self.viz = viz
        self.env =  gym.make('CartPole-v0')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state = None

        # For counting the max-timestep
        self.timestep = 0

    def reset(self):
        self.state = self.env.reset()
        return self.current_state()

    def current_state(self):
        return self.state
    
    def render(self):
        self.env.render()
    
    def step(self, a):
        if self.viz:
            self.env.render()
        next_state, reward, done, info = self.env.step(a)
        self.state = next_state
        self.timestep += 1

        if done:
            if self.timestep < self.MAX_TIMESTEP:
                reward = -20
            self.env.reset()
            self.timestep = 0
        return next_state, reward, done, info
      
        
