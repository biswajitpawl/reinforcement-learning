import numpy as np

def argmax(values):
    max_val = values[0]
    best_actions = [0]
    for i in range(1, len(values)):
        if values[i] > max_val:
            max_val = values[i]
            best_actions = [i]
        elif values[i] == max_val:
            best_actions.append(i)
    return np.random.choice(best_actions)

def softmax(vals):
    max_val = np.amax(vals)
    exps = np.exp(vals - max_val)
    return exps / np.sum(exps)