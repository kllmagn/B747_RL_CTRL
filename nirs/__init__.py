import gym

gym.register(
     id='CtrlEnv-v0',
     entry_point='project.nirs.envs:ControllerEnv'
)