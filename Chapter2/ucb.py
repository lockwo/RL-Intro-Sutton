import gym 
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import random 
from tqdm import tqdm

def get_action(Q, e):
    if random.random() < e:
        return random.randint(0, 9)
    else:
        return np.argmax(Q)

def average(tot_rewards):
    avg = []
    for i in range(len(tot_rewards[0])):
        temp = 0
        for j in range(len(tot_rewards)):
            temp += tot_rewards[j][i]
        temp /= len(tot_rewards)
        avg.append(temp)
    return avg

def ucb(Q, N, p):
    vals = Q + np.sqrt(-np.log(p) / (2 * N + 1e-8))
    return np.argmax(vals)

def ucb1(Q, N, t):
    vals = Q + np.sqrt((2 * np.log(t)) / (N + 1e-8))
    return np.argmax(vals)

k = 10 
iter = 1000
repeat = 2000
u0 = []
u1 = []
e01 = []
env = gym.make("BanditTenArmedGaussian-v0")

p = 0.1
env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    Q = np.zeros(shape=(k))
    N = np.zeros(shape=(k))
    rewards = []
    for i in range(iter):
        action = ucb(Q, N, p)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        N[action] += 1
        Q[action] = Q[action] +  1/N[action] * (reward - Q[action])
    env.reset()
    tot_rewards.append(rewards)

u0 = average(tot_rewards)

env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    Q = np.zeros(shape=(k))
    N = np.zeros(shape=(k))
    rewards = []
    for i in range(iter):
        action = ucb1(Q, N, i + 1)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        N[action] += 1
        Q[action] = Q[action] +  1/N[action] * (reward - Q[action])
    env.reset()
    tot_rewards.append(rewards)

u1 = average(tot_rewards)

env.reset()
tot_rewards = []
e = 0.1
for _ in tqdm(range(repeat)):
    Q = np.zeros(shape=(k))
    N = np.zeros(shape=(k))
    rewards = []
    for i in range(iter):
        action = get_action(Q, e)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        N[action] += 1
        Q[action] = Q[action] +  1/N[action] * (reward - Q[action])
    env.reset()
    tot_rewards.append(rewards)

e01 = average(tot_rewards)

plt.plot(u0, color='green', label='UCB')
plt.plot(u1, color='blue', label='UCB1')
plt.plot(e01, color='olive', label='e = 0.1')
plt.xlabel("Steps")
plt.ylim(0, max(u0) + 0.2)
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Steps on 10 Armed Bandit")
plt.legend()
plt.show()
