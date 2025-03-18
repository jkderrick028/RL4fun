import numpy as np
import matplotlib.pyplot as plt


def transition(s, a):
    """
    from state s, take action a, return s' and r

    Arguments:
        s: int tuple, (row, col)
        a: str, 'l', 'r', 'u', 'd'
    Returns
        s_prime: int tuple
        r: float, reward
    """

    if a == 'u':
        s_prime = (np.max([s[0] - 1, 0]), s[1])
    elif a == 'd':
        s_prime = (np.min([s[0] + 1, n_rows - 1]), s[1])
    elif a == 'l':
        s_prime = (s[0], np.max([s[1] - 1, 0]))
    else:
        s_prime = (s[0], np.min([s[1] + 1, n_cols - 1]))

    if gridworld[s_prime] == -1:
        r = -100
    else:
        r = -1
    return s_prime, r




np.random.seed(2025)

n_rows = 4
n_cols = 12

gridworld = np.zeros((n_rows, n_cols))
gridworld[3, 0] = 1     # start
gridworld[3, -1] = 2    # end
gridworld[3, 1:-1] = -1 # cliff

actions = ['u', 'd', 'l', 'r']
Qs = {(r, c):np.zeros(len(actions)) for r in np.arange(n_rows) for c in np.arange(n_cols)}
epsilon = 0.1

# Q-learning
n_episodes = 500
alpha = 0.5
sum_of_rewards = []

for epiI in np.arange(n_episodes):
    s = (n_rows-1, 0)
    sum_r = 0
    record = []
    while gridworld[s] != 2:
        rand = np.random.rand()
        if rand > epsilon:
            aI = np.argmax(Qs[s])
        else:
            aI = np.random.randint(0, len(actions))

        # aI = np.argmax(Qs[s])
        a = actions[aI]
        s_prime, r = transition(s, a)

        Qs[s][aI] = Qs[s][aI] + alpha * (r + np.max(Qs[s_prime]) - Qs[s][aI])
        sum_r += r

        record.append(s)
        record.append(a)
        record.append(r)

        if gridworld[s_prime] == -1:
            break
            # s = (n_rows-1, 0)
        else:
            s = s_prime

    sum_of_rewards.append(sum_r)
    print()


plt.plot(sum_of_rewards)
plt.show()
pass
