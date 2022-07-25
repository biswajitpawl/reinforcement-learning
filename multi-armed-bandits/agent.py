import numpy as np
from utils import argmax, softmax

class Agent:
    
    def __init__(self, arms, strategy, init_val=0., **params):
        self.arms = arms
        self.strategy = strategy
        self.init_val = init_val
        self.params = params
        
        self.estimates = np.full(arms, init_val)
        self.action_counts = np.zeros(arms)
        self.prob = None
        self.rewards_sum = 0.

    def pull(self):
        if self.strategy == 'epsilon_greedy':
            epsilon = self.params['epsilon']
            if np.random.random_sample() < epsilon:
                action = np.random.randint(self.arms) # Explore
            else:
                action = argmax(self.estimates) # Exploit
        
        elif self.strategy == 'ucb':
            c = self.params['c']
            if 0 in self.action_counts:
                unused_arms = np.where(self.action_counts == 0)[0]
                action = np.random.choice(unused_arms)
            else:
                total_steps = np.sum(self.action_counts)
                ucb_terms = self.estimates + c * np.sqrt(np.log(total_steps)/self.action_counts)
                action = argmax(ucb_terms)
                
        elif self.strategy == 'gradient':
            self.prob = softmax(self.estimates)
            action = np.random.choice(self.arms, p=self.prob)
        
        self.action_counts[action] += 1
        return action

    def update_estimate(self, reward, action):
        if self.strategy in ['epsilon_greedy', 'ucb']:
            step_size = self.params['step_size'] if 'step_size' in self.params else 1./self.action_counts[action]
            self.estimates[action] += step_size * (reward - self.estimates[action])
            
        elif self.strategy == 'gradient':
            step_size = self.params['step_size']
            delta = reward
            baseline = self.params['baseline'] if 'baseline' in self.params else True
            if baseline:
                self.rewards_sum += reward
                total_steps = np.sum(self.action_counts)
                delta -= self.rewards_sum / total_steps

            self.estimates[action] += step_size * delta * (1. - self.prob[action])
            self.estimates[:action] -= step_size * delta * self.prob[:action]
            self.estimates[action+1:] -= step_size * delta * self.prob[action+1:]

    def reset(self):
        self.estimates[:] = self.init_val
        self.rewards_sum = 0.
        self.action_counts[:] = 0