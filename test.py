from math import pi
import os
import sys
import gym

from elegantrl.train.run import train_and_evaluate
from elegantrl.train.evaluator import get_episode_return_and_step
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.utils import init_agent
import numpy as np
import torch

from core.controller import CtrlMode, CtrlType, ResetRefMode
from env.ctrl_env import ControllerEnv, ObservationType, RewardType

env_args={
    'max_step': 400,
    'env_num': 1,
    'env_name': 'ControllerEnv',
    'state_dim': 3,
    'action_dim': 1,
    'if_discrete': False,
    'target_return': 350,
}

dir_path = './ControllerEnv_PPO_-1'

if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    agent_class = AgentPPO

    env = lambda: ControllerEnv(
        ObservationType.PID_LIKE,
        RewardType.CLASSIC,
        True,
        True,
        CtrlType.MANUAL,
        CtrlMode.DIRECT_CONTROL,
        tk = 20,
        reset_ref_mode = ResetRefMode.CONST,
        disturbance_mode = None,
        sample_time = 0.05,
        use_limiter = False,
        action_max=17*pi/180,
        aero_err=None,
    )
    args = Arguments(agent_class, env_func=env, env_args=env_args)
    #args.reward_scale = 1  # RewardRange: -1800 < -200 < -50 < 0
    args.gamma = 0.97
    args.target_step = args.max_step * 8
    args.eval_times = 2 ** 3
    args.learner_gpus = -1
    #args.random_seed 

    #train_and_evaluate(args)

    agent = init_agent(args, gpu_id=args.learner_gpus, env=env)
    act, b, c, d, e, f = agent.save_or_load_agent(args.cwd, False)
    act.load_state_dict(input("Путь к файлу: "))

    env_test = env()
    env_test.ctrl.use_storage = True
    s = env.reset()
    while True:
        action = act.get_action(torch.tensor(s))
        next_state, reward, done, _ = env_test.step(action.detach().numpy())
        if done:
            s = env.reset()
            break
        state = next_state
        env.render()
    env_test.ctrl.storage.plot(["vartheta", "vartheta_ref"], base="t")