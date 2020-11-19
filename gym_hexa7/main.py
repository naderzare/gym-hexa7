import gym

env = gym.make('gym_hexa7:hexa7-v0')
obs = env.reset()
done = False
while not done:
    env.render()
    print(env.possible_action())
    obs, reward, done, info = env.step((int(input('number:')) - 1) * 6 + int(input('dir:')))
