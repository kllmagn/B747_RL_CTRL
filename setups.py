from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
from callbacks import *

from typing import Callable

import torch as th

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

hyperparams = {
    SAC: {
        #'tau': 0.995,
        'buffer_size': int(5e5),
        #'learning_rate': linear_schedule(4e-4),
        #'policy_kwargs': dict(net_arch=[64, 64]),
        #'target_entropy': -3,
        'learning_starts': int(10000)
    },
    PPO: {
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[321, 210, 206]),
        'learning_rate': 0.0008119972057477548,
        'gae_lambda': 0.9299298432752194,
        'n_steps': 155,
        'ent_coef': 0.005236684910302408,
        'vf_coef': 0.46999041106889117,
        'batch_size': 155, #120,
        'gamma': 0.8351629380091844
        },
    TD3: {
        'gamma': 0.9999,
        'learning_rate': 0.00045845313560993127,
        'batch_size': 64,
        'buffer_size': 100000,
        'tau': 0.01,
        'train_freq': 1,
        #'noise_type': None,
        #'noise_std': 0.37398891007928636,
        'policy_kwargs': dict(net_arch=[400, 300])
        },
    'A2C_bad': {
        #'gamma': 0.7499538451822588,
        #'max_grad_norm': 0.5843626531806608,
        #'use_rms_prop': False,
        #'gae_lambda': 0.8669317637280056,
        #'n_steps': 124,
        #'learning_rate': 4.903911208190676e-05,
        #'ent_coef': 0.006097913388504438,
        #'vf_coef': 0.1920274381094792,
        'policy_kwargs': dict(ortho_init=True, activation_fn=th.nn.Tanh, net_arch=[300, 300])
        },
    'A2C_repl': {
        'gamma': 0.8909336106571547,
        'max_grad_norm': 0.6649707879260869,
        'use_rms_prop': True,
        'gae_lambda': 0.8551027353954989,
        'n_steps': 240,
        'learning_rate': 0.0008179783358248342,
        'ent_coef': 0.005168290493653929,
        'vf_coef': 0.19324489075054455,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[359, 390])
    },
    'A2C_backup':  {
        'gamma': 0.943688888281661,
        'max_grad_norm': 0.5409540085484815,
        'use_rms_prop': False,
        'gae_lambda': 0.6241936925284,
        'n_steps': 251,
        'learning_rate': 0.0009392966757968232,
        'ent_coef': 0.006226426696876032,
        'vf_coef': 0.47678293409139105,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[339, 368, 331])
    },
    A2C: {
        #'use_sde': True,
        'use_rms_prop': True,
        'learning_rate': 0.00013219127332957597,
        'ent_coef': 0.0026650043954570186,
        'vf_coef': 0.10796014008883446,
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[337, 380]) #, optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
        },
    'A2C_old': {
        'gamma': 0.95,
        'normalize_advantage': True,
        'max_grad_norm': 0.8,
        'use_rms_prop': True,
        'gae_lambda': 0.92,
        'n_steps': 128,
        'learning_rate': linear_schedule(0.0006784691121472965),
        'ent_coef': 0.0026650043954570186,
        'vf_coef': 0.10796014008883446,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[260, int((26*10)**0.5), 10, 1], vf=[260,int((260*5)**0.5), 5, 1])]) #dict(pi=[254,254], vf=[254,254])])
    },
    'A2C_old1': {
        'gamma': 0.95,
        #'normalize_advantage': True,
        'max_grad_norm': 0.8,
        'use_rms_prop': False,
        'gae_lambda': 0.92,
        'n_steps': 128,
        'learning_rate': linear_schedule(0.0006784691121472965),
        'ent_coef': 0.0026650043954570186,
        'vf_coef': 0.10796014008883446,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])]) #dict(pi=[254,254], vf=[254,254])])
    },
    'A2C_1': {
        'gamma': 0.9,
        'max_grad_norm': 0.8,
        'use_rms_prop': False,
        'gae_lambda': 0.9,
        'n_steps': 8,
        'learning_rate': 1.0402502087400854e-05,
        'ent_coef': 0.021529322791225745,
        'vf_coef': 0.4844797321283937,
        'policy_kwargs': dict(ortho_init=True, activation_fn=th.nn.Tanh, net_arch=[348, 354])
    }
}