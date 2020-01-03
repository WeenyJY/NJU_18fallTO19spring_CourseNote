from lib.envs.gridworld import GridworldEnv
# initialize
env = GridworldEnv()
# render env
env._render()

print('State space:', env.nS)
print('Action space:', env.nA)
# P[state][action]
# return: probability, next_state, reward, is_terminated
print('Action space:', env.P[14][3])
