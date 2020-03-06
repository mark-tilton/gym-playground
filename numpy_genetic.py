import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')

observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

w1 = np.zeros((observation_size, action_size))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def simulate(n, w, render):
    total_reward = 0.0
    observation = env.reset()
    for _ in range(n):
        if render:
            env.render()
        y = np.dot(observation, w)
        # y = softmax(y)
        # a = np.random.choice(y, p=y)
        a = np.argmax(y)
        observation, reward, done, _ = env.step(a)
        total_reward += reward
        if done == True:
            break
    return total_reward


x = []
y = []
total_eps = 0
training_steps = 10000
learning_rate = 1e-3
max_frames = 250
for i in range(training_steps):
    batch_size = 5
    avg_score = 0
    for _ in range(batch_size):
        avg_score += simulate(max_frames, w1, False)
        total_eps += 1
    avg_score /= batch_size
    x.append(total_eps)
    y.append(avg_score)
    print(f'{round(i / (training_steps - 1) * 100)}% | {total_eps} | {avg_score}')
    w1gds = []
    batch_size = 10
    for _ in range(batch_size):
        w1gd = np.random.randn(observation_size, action_size)
        nw1 = w1 + w1gd
        reward = simulate(max_frames, nw1, False)
        total_eps += 1
        delta_r = reward - avg_score
        w1gds.append(w1gd * delta_r)
    w1 = w1 + sum(w1gds) * learning_rate / batch_size

env.close()

plt.plot(x, y)
plt.show()

while True:
    simulate(max_frames, w1, True)
