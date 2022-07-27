import numpy as np
from tqdm import tqdm

class Environment:

    def __init__(self, pulls, iterations, testbed):
        self.pulls = pulls
        self.iterations = iterations
        self.testbed = testbed

    def run(self, agent, verbose=True):
        scores_avg = []
        optimal_counts = np.zeros(self.pulls)
        
        # Logger
        if verbose:
            msg = f'{agent.strategy} agent: '
            for k, v in agent.params.items():
                msg += f'{k}={v}, '
            msg += f'init_val={agent.init_val}, testbed={"stationary." if self.testbed.stationary else "non-stationary."}'
            print(msg + '\n' + '-' * len(msg))
    
        for i in tqdm(range(self.iterations), disable=not(verbose)):
            # Set a random seed for each iteration/experiment
            np.random.seed(i)
            
            # Reset the testbed and agent
            self.testbed.reset()
            agent.reset()
            
            # Choose actions through time-steps / Pull arms
            scores = []
            for p in range(self.pulls):
                # Pull an arm > get reward > update value-estimate
                action = agent.pull()
                reward = self.testbed.get_reward(action)
                agent.update_estimate(reward, action)

                # Store reward at each time-step/pull
                scores.append(reward)
                
                # Store total no. of optimal actions
                if action == self.testbed.optimal_action:
                    optimal_counts[p] += 1
                    
            scores_avg.append(scores)
                
        print()
        
        scores_avg = np.mean(scores_avg, axis=0)
        optimal_counts /= self.iterations
        
        return scores_avg, optimal_counts*100
    
    def reset(self, testbed):
        self.testbed = testbed