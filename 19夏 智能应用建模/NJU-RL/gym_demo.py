import gym

env = gym.envs.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))
print(env.unwrapped.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

# random action for 1000 steps
for _ in range(10000):
  env.render()
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  # reset if terminated
  if done:
    observation = env.reset()
env.close()