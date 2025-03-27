import numpy as np
import matplotlib.pyplot as plt


def transition(s, a):
    x, y = s
    if a == 'u':
        x = np.max([0, x - 1 - wind_strengths[y]])
    elif a == 'd':
        x_prime = x + 1 - wind_strengths[y]
        x = np.max([0, x_prime])
        x = np.min([x, n_rows - 1])
    elif a == 'l':
        x = np.max([0, x - wind_strengths[y]])
        y = np.max([0, y - 1])
    elif a == 'r':
        x = np.max([0, x - wind_strengths[y]])
        y = np.min([y + 1, n_cols - 1])

    return (x, y)


np.random.seed(2025)
n_rows = 7
n_cols = 10
start = (3, 0)
goal = (3, 7)
wind_strengths = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

s_prime = transition((1, 7), 'r')
print(s_prime)

epsilon = 0.1
alpha = 0.5
n_episodes = 100

actions = ['u', 'd', 'l', 'r']
Qs = {
    (x, y): np.zeros(len(actions)) for x in np.arange(n_rows) for y in np.arange(n_cols)
}

n_steps = 0
counts = np.zeros(n_episodes)

for epI in np.arange(n_episodes):
    if np.mod(epI, 2) == 0:
        print(epI)
    s = start
    randnum = np.random.rand()
    if randnum > epsilon:
        vals = Qs[s]
        aI = np.random.choice([action_ for action_, value_ in enumerate(vals) if value_ == np.max(vals)])
    else:
        aI = np.random.randint(len(actions))
    a = actions[aI]

    while s != goal:
        s_prime = transition(s, a)
        r = -1

        randnum = np.random.rand()
        if randnum > epsilon:
            vals = Qs[s_prime]
            aI_prime = np.random.choice([action_ for action_, value_ in enumerate(vals) if value_ == np.max(vals)])
        else:
            aI_prime = np.random.randint(len(actions))
        a_prime = actions[aI_prime]

        Qs[s][aI] += alpha * (r + Qs[s_prime][aI_prime] - Qs[s][aI])

        s = s_prime
        a = a_prime

        n_steps += 1

    counts[epI] = n_steps



