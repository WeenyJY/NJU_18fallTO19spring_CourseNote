import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../")

from lib.envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()
# always start at grid 36
print(env.reset())
env.render()
# grid no., reward, is_terminated, Prob
print(env.step(0))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()