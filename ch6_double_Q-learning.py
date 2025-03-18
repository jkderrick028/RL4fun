import numpy as np
import matplotlib.pyplot as plt


def transition(s, a):
    if s == 'A':
        if a == 'R':
            r = 0
            s = 'T'
        else:
            s = 'B'
            r = 0
    else:
        r = np.random.randn() - 0.1
        s = 'T'

    return s, r


def action(s):
    randnum = np.random.uniform()
    actions = list(Qs[s].keys())

    if randnum > epsilon:
        a = actions[np.argmax(Qs[s])]
    else:
        a = actions[np.random.randint(low=0, high=len(Qs[s]))]
    return a


states = ['A', 'B']
epsilon = 0.1
alpha = 0.1
np.random.seed(2025)

n_actions_B = 10

Qs = {
    'A': {'L': 0.5, 'R': 0.5},
    'B': {f'a{k}': 1/n_actions_B for k in np.arange(n_actions_B)},
    'T': 0
}

# Q-learning

n_episodes = 100
for episodeI in np.arange(n_episodes):
    s = 'A'

    while s != 'T':
        a = action(s)
        s_prime, r = transition(s, a)
        if s_prime == 'T':
            Qs[s][a] += alpha * (r + 0 - Qs[s][a])
        else:
            Qs[s][a] += alpha * (r + np.max(list(Qs[s_prime].values())) - Qs[s][a])

        s = s_prime

