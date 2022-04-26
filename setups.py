from matplotlib.style import use
from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from callbacks import *

from typing import Callable

import torch as th

from optuna import Trial

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[337, 380]) #, optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-7))
        },
    "A2Copt": {
        'gamma': 0.99,
        'max_grad_norm': 0.9,
        'use_rms_prop': False,
        'gae_lambda': 0.99,
        'n_steps': 1, # выбрано
        'learning_rate': 0.0038416303921840625,
        'ent_coef': 1.2239639220509283e-06,
        'vf_coef': 0.3489087299420313,
        'policy_kwargs': dict(ortho_init=True, activation_fn=th.nn.Tanh, net_arch=[346, 346]),
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

def trial_hyperparams(net_class, trial:Trial, default_hp:dict={}):
    if net_class is A2C:
        hp = {
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        'max_grad_norm': trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
        'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
        'gae_lambda': trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        'n_steps': trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
        'ent_coef': trial.suggest_loguniform("ent_coef", 0.00000001, 0.1),
        'vf_coef': trial.suggest_uniform("vf_coef", 0, 1),
        'policy_kwargs': dict(ortho_init=trial.suggest_categorical("ortho_init", [False, True]), activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 300, 400) for i in range(trial.suggest_int('n_depth', 2, 3))]) #dict(pi=[254,254], vf=[254,254])])
        }
    elif net_class is TD3:
        hp = {
        'gamma': trial.suggest_float('gamma', 0.7, 0.99),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'buffer_size': trial.suggest_int('buffer_size', 20000, 200000),
        'tau': trial.suggest_float('tau', 0.001, 0.1),
        'train_freq': trial.suggest_int('train_freq', 1, 64),
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 32, 400) for i in range(trial.suggest_int('n_depth', 2, 4))])
        }
    elif net_class is PPO:
        hp = {
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 32, 400) for i in range(trial.suggest_int('n_depth', 2, 4))]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.6, 1.0),
        'n_steps': trial.suggest_int('n_steps', 1, 256),
        'ent_coef': trial.suggest_float('ent_coef', 0, 0.01),
        'vf_coef': trial.suggest_float('vf_coef', 0.05, 0.6),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'gamma': trial.suggest_float('gamma', 0.7, 0.99),
        }
    elif net_class is SAC:
        hp = {
        'use_sde': trial.suggest_categorical('use_rms_prop', [True, False]),
        'gamma': trial.suggest_float('gamma', 0.7, 0.99),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'buffer_size': trial.suggest_int('buffer_size', 20000, 200000),
        'tau': trial.suggest_float('tau', 0.001, 0.1),
        'train_freq': trial.suggest_int('train_freq', 1, 64),
        'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 32, 400) for i in range(trial.suggest_int('n_depth', 2, 4))])
        }
    else:
        hp = default_hp
    return hp


class TrainPlotter:
    def __init__(self, env, y_labels:list, x_label:str):
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1,1,1)
        self.ax1.grid(True)
        #self.ax1.set_xlim(0, 20)
        self.ax1.set_xticks([0, 5, 10, 15, 20])
        self.xs = []
        self.x_label = x_label
        self.ys = dict([(y_label, []) for y_label in y_labels])
        self.y_labels = y_labels
        self.lines = dict([(k, self.ax1.plot(self.xs, v, label=k)[0]) for k, v in self.ys.items()])
        self.ax1.legend()
        self.env = env
        self.x_len = 0
        self.storage = None
        self.an = animation.FuncAnimation(fig, self._animate, interval=100)
    
    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def _animate(self, _):
        if self.storage is None:
            storage = self.env.get_attr('ctrl')[0].storage.storage
        else:
            storage = self.storage
        self.xs = storage[self.x_label] if self.x_label in storage else []
        if self.x_len > len(self.xs):
            self.ax1.set_xlim(0, 0)
            self.ax1.set_ylim(0, 0)
        self.x_len = len(self.xs)
        for y_label in self.y_labels:
            self.ys[y_label] = storage[y_label] if y_label in storage else []
        common_len = min([len(self.xs)] + [len(y) for y in self.ys.values()])
        x = self.xs[:common_len]
        for y_label in self.y_labels:
            y = self.ys[y_label][:common_len]
            self.lines[y_label].set_xdata(x)
            self.lines[y_label].set_ydata(y)
            xlim, ylim = self.ax1.get_xlim(), self.ax1.get_ylim()
            if len(x):
                self.ax1.set_xlim(min(xlim[0], min(x)), max(xlim[1], max(x)))
            if len(y):
                self.ax1.set_ylim(min(ylim[0], min(y)), max(ylim[1], max(y)))
        return [self.lines[y_label] for y_label in self.y_labels]