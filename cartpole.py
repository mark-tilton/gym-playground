import gym
import numpy as np

env = gym.make('CartPole-v0')

observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

w1 = np.random.randn(observation_size, action_size)


def simulate(n, w, render):
    total_reward = 0.0
    observation = env.reset()
    for _ in range(n):
        if render:
            env.render()
        y = np.dot(observation, w)
        action = np.argmax(y)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done == True:
            break
    return total_reward


training_steps = 10000
learning_rate = 1e-3
for i in range(training_steps):
    base_reward = simulate(1000, w1, i % 10 == 9)
    w1gds = []
    batch_size = 10
    for _ in range(batch_size):
        w1gd = np.random.randn(observation_size, action_size)
        b1gd = np.random.randn(action_size)
        nw1 = w1 + w1gd
        total_reward = 0.0
        num_steps = 1000
        reward = simulate(num_steps, nw1, False)
        delta_r = reward - base_reward
        w1gds.append(w1gd * delta_r)
    w1 = w1 + sum(w1gds) * learning_rate / batch_size

env.close()
