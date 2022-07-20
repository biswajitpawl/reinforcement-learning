import numpy as np

class Testbed:
    
    def __init__(self, arms, stationary=True):
        self.arms = arms
        self.stationary = stationary
        self.reset()

    def get_reward(self, action):
        if not self.stationary and self.count == 500: self.reset()
        self.count += 1
        return np.random.normal(self.q_actuals[action], 1.)

    def reset(self):
        self.count = 0
        self.q_actuals = np.random.randn(self.arms)
        self.q_optimal = np.amax(self.q_actuals)
        self.optimal_action = np.argmax(self.q_actuals)