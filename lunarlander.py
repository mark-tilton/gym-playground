import gym
import numpy as np

env = gym.make('LunarLander-v2')

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
render_frequency = 10
max_frames = 250
for i in range(training_steps):
    batch_size = 10
    avg_score = 0
    for _ in range(batch_size):
        avg_score += simulate(max_frames, w1, False)
    avg_score /= batch_size
    if i % render_frequency == 0:
        simulate(max_frames, w1, True)
        print(avg_score)
    w1gds = []
    for _ in range(batch_size):
        w1gd = np.random.randn(observation_size, action_size)
        nw1 = w1 + w1gd
        total_reward = 0.0
        reward = simulate(max_frames, nw1, False)
        delta_r = reward - avg_score
        w1gds.append(w1gd * delta_r)
    w1 = w1 + sum(w1gds) * learning_rate / batch_size

env.close()
