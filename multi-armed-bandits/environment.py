import numpy as np

class Environment:

    def __init__(self, pulls, iterations, testbed):
        self.pulls = pulls
        self.iterations = iterations
        self.testbed = testbed

    def run(self, agent):
        scores_avg = []
        optimal_counts = np.zeros(self.pulls)
        
        # Logger
        msg = f'{agent.strategy} agent | params: {agent.params}'
        print(msg + '\n' + '-' * len(msg))
    
        for i in range(self.iterations):
            # Set a random seed for each iteration/experiment
            np.random.seed(i)
            
            # Reset the testbed and agent
            self.testbed.reset()
            agent.reset()
            
            # Choose actions through time-steps / Pull arms
            # rewards_sum = 0.
            scores = []
            for p in range(self.pulls):
                # Pull an arm > get reward > update value-estimate
                action = agent.pull()
                reward = self.testbed.get_reward(action)
                agent.update_estimate(reward, action)

                # Store avg. reward at each time-step/pull
                # rewards_sum += reward
                scores.append(reward) #(rewards_sum/(p+1))
                
                # Store total no. of optimal actions chosen (over iterations/experiments) at each time-step/pull
                if action == self.testbed.optimal_action:
                    optimal_counts[p] += 1
                    
            scores_avg.append(scores)
            
            if i%100 == 99 or i == self.iterations-1:
                print(f'No. of iterations completed: {i+1}')
                
        print()
        
        scores_avg = np.mean(scores_avg, axis=0)
        optimal_counts /= self.iterations
        
        return scores_avg, optimal_counts
    
    def reset(self, testbed):
        self.testbed = testbed