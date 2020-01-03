import gym

env = gym.envs.make("SpaceInvaders-v0")

print("Action space size: {}".format(env.action_space.n))

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

# random action for 1000 steps
for idx in range(10000):
  env.render()
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  if idx == 0:
      print("states demo:", observation)
      print("action demo:", action)
      print("reward demo:", reward)

  # reset if terminated
  if done:
    observation = env.reset()
env.close()