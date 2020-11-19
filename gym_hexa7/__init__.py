from gym.envs.registration import register

register(
    id='hexa7-v0',
    entry_point='gym_hexa7.envs:Hexa7Env',
)
