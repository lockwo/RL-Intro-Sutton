import gym 
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#def softmax(h):
#    return np.exp(h)/sum(np.exp(h))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_action(h):
    probs = softmax(h)
    return np.random.choice(10, p=probs), probs

def average(tot_rewards):
    avg = []
    for i in range(len(tot_rewards[0])):
        temp = 0
        for j in range(len(tot_rewards)):
            temp += tot_rewards[j][i]
        temp /= len(tot_rewards)
        avg.append(temp)
    return avg

k = 10 
iter = 1000
repeat = 2000
o0 = []
o00 = []
e00 = []
e01 = []
env = gym.make("BanditTenArmedGaussian-v0")

alpha = 0.4

env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    H = np.zeros(shape=(k))
    rewards = []
    for i in range(iter):
        action, probs = get_action(H)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        H[action] = H[action] + alpha * reward * (1 - probs[action])
        H[:action] = H[:action]  - alpha * reward * probs[:action]
        if action + 1 < k:
            H[action + 1:] = H[action + 1:]  - alpha * reward * probs[action + 1]
    env.reset()
    tot_rewards.append(rewards)

o0 = average(tot_rewards)

env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    H = np.zeros(shape=(k))
    R = 0
    rewards = []
    for i in range(iter):
        action, probs = get_action(H)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        H[action] = H[action] + alpha * (reward - R) * (1 - probs[action])
        H[:action] = H[:action]  - alpha * (reward - R) * probs[:action]
        if action + 1 < k:
            H[action + 1:] = H[action + 1:]  - alpha * (reward - R) * probs[action + 1]
        R = R + 1/(i + 1) * (reward - R)
    env.reset()
    tot_rewards.append(rewards)

e01 = average(tot_rewards)

alpha = 0.1

env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    H = np.zeros(shape=(k))
    rewards = []
    for i in range(iter):
        action, probs = get_action(H)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        H[action] = H[action] + alpha * reward * (1 - probs[action])
        H[:action] = H[:action]  - alpha * reward * probs[:action]
        if action + 1 < k:
            H[action + 1:] = H[action + 1:]  - alpha * reward * probs[action + 1]
    env.reset()
    tot_rewards.append(rewards)

o00 = average(tot_rewards)

env.reset()
tot_rewards = []
for _ in tqdm(range(repeat)):
    H = np.zeros(shape=(k))
    R = 0
    rewards = []
    for i in range(iter):
        action, probs = get_action(H)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        H[action] = H[action] + alpha * (reward - R) * (1 - probs[action])
        H[:action] = H[:action]  - alpha * (reward - R) * probs[:action]
        if action + 1 < k:
            H[action + 1:] = H[action + 1:]  - alpha * (reward - R) * probs[action + 1]
        R = R + 1/(i + 1) * (reward - R)
    env.reset()
    tot_rewards.append(rewards)

e00 = average(tot_rewards)

plt.plot(o0, color='blue', label='No Baseline, a = 0.4')
plt.plot(e01, color='olive', label='Baseline, a = 0.4')
plt.plot(o00, color='red', label='No Baseline, a = 0.1')
plt.plot(e00, color='green', label='Baseline, a = 0.1')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Steps on 10 Armed Bandit")
plt.legend()
plt.show()