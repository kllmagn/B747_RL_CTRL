import os
import numpy as np
from math import pi

from nirs.envs.ctrl_env.ctrl_env import ControllerEnv
from pretrain import pretrain_agent

from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.a2c.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from callbacks import *
from tensorboard import program

from typing import Callable

from tqdm import tqdm

import torch as th

import optuna

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

hyperparams = {
    PPO: {

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
    A2C: {
        'seed': 1,
        'gamma': 0.8591592569806266,
        'max_grad_norm': 0.7117778868069197,
        'use_rms_prop': False,
        'gae_lambda': 0.9743463632001759,
        'n_steps': 185,
        'learning_rate': 0.0005211215934042513,
        'ent_coef': 0.00022920566926045685,
        'vf_coef': 0.07751990459360632,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[104, 362, 141])
    },
    'A2C_base': {
        'gamma': 0.95,
        #'normalize_advantage': True,
        'max_grad_norm': 0.8,
        'use_rms_prop': True,
        'gae_lambda': 0.92,
        'n_steps': 128,
        'learning_rate': linear_schedule(0.0006784691121472965),
        'ent_coef': 0.0026650043954570186,
        'vf_coef': 0.10796014008883446,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[260, int((26*10)**0.5), 10, 1], vf=[260,int((260*5)**0.5), 5, 1])]) #dict(pi=[254,254], vf=[254,254])])
    },
    'A2C_old': {
        'gamma': 0.95,
        #'normalize_advantage': True,
        'max_grad_norm': 0.8,
        'use_rms_prop': False,
        'gae_lambda': 0.92,
        'n_steps': 128,
        'learning_rate': linear_schedule(0.0006784691121472965),
        'ent_coef': 0.0026650043954570186,
        'vf_coef': 0.10796014008883446,
        'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64,64], vf=[64,64])]) #dict(pi=[254,254], vf=[254,254])])
    }
}

class ControllerAgent:
    def __init__(self, net_class=A2C, use_tb=False, log_dir='./logs'):
        self.log_dir = log_dir # папка с логами
        self.tb_log = os.path.join(self.log_dir, 'tb_log') if use_tb else None
        if use_tb:
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_log])
            self.tb_url = self.tb.launch()
            print(f"TensorBoard listening on {self.tb_url}")
        else:
            self.tb, self.tb_url = None, None
        self.net_class = net_class
        self.hp = hyperparams[self.net_class] if self.net_class in hyperparams else {}
        self.model = None
        self.bm_name = 'best_model.zip'

    def _wrap_env(self, env):
        env = Monitor(env, os.path.join(self.log_dir, 'monitor.csv'))
        env = DummyVecEnv([lambda: env])
        #env = VecNormalize(env, gamma=0.95, norm_reward=False)
        env.seed(1)
        return env

    def _unwrap_env(self, env):
        return env.envs[0].env

    def optimize(self, training_timesteps, *ctrl_env_args, pretrain=False, **ctrl_env_kwargs):
        def save_model_callback(study:optuna.Study, trial):
            if study.best_value <= trial.value:
                save_path = os.path.join(self.log_dir, 'best_model.zip')
                print(f"Saving new best trial to {save_path}")
                self.model.save(save_path)
                env = self.model.env
                self.test_env(5000-1, env)
                env.envs[0].env.ctrl.storage.save(base='t')
                env.envs[0].env.ctrl.storage.plot(['vartheta', 'vartheta_ref'], 't')
        def objective(trial:optuna.Trial):
            if self.net_class is A2C:
                hp = {
                'seed': 1,
                'gamma': trial.suggest_float('gamma', 0.7, 0.99),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 0.8),
                'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.6, 1.0),
                'n_steps': trial.suggest_int('n_steps', 1, 256),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
                'ent_coef': trial.suggest_float('ent_coef', 0, 0.01),
                'vf_coef': trial.suggest_float('vf_coef', 0.05, 0.6),
                'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 32, 400) for i in range(trial.suggest_int('n_depth', 2, 4))]) #dict(pi=[254,254], vf=[254,254])])
                }
            elif self.net_class is TD3:
                hp = {
                'gamma': trial.suggest_float('gamma', 0.7, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
                'batch_size': trial.suggest_int('batch_size', 32, 256),
                'buffer_size': trial.suggest_int('buffer_size', 20000, 200000),
                'tau': trial.suggest_float('tau', 0.001, 0.1),
                'train_freq': trial.suggest_int('train_freq', 1, 64),
                'policy_kwargs': dict(activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 32, 400) for i in range(trial.suggest_int('n_depth', 2, 4))])
                }
            elif self.net_class is PPO:
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
            else:
                hp = self.hp
            env = ControllerEnv(*ctrl_env_args, use_storage=True, **ctrl_env_kwargs)
            env = self._wrap_env(env)
            cb = EarlyStopping(10000, 4, self.log_dir, verbose=1)
            self.model = self.net_class('MlpPolicy', env, verbose=0, tensorboard_log=self.tb_log, **hp)
            if pretrain:
                env_expert = DummyVecEnv([lambda: ControllerEnv(manual_control=False)])
                self.model = pretrain_agent(self.model, env_expert, return_policy=False, verbose=0, num_interactions=20e3)
            total_timesteps = training_timesteps
            with ProgressBarManager(total_timesteps) as callback:
                cb_list = CallbackList([callback, cb])
                self.model.learn(total_timesteps=total_timesteps, callback=cb_list)
            mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=10, render=True)
            print(f"Mean reward = {mean_reward} +/- {std_reward}")
            return mean_reward-std_reward
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, callbacks=[save_model_callback])
        params = study.best_params
        print('Лучшие параметры:', params)
        params['policy_kwargs'] = dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[params[f'n{i+1}'] for i in range(params['n_depth'])])
        for i in range(params['n_depth']):
            del params[f'n{i+1}'] 
        del params['n_depth']
        self.hp = params
        print('Параметры нейросети:', self.hp)

    def train(self, *ctrl_env_args, timesteps=50000, pretrain=True, optimize=False, **ctrl_env_kwargs):
        if optimize:
            print('Оптимизирую с помощью Optuna')
            self.optimize(100000, *ctrl_env_args, pretrain=pretrain, **ctrl_env_kwargs)
            self.model.load(os.path.join(self.log_dir, 'best_model.zip'))
        else:
            self.model = self.net_class('MlpPolicy', verbose=1, tensorboard_log=self.tb_log, **self.hp)
        self.env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(self.env)
        self.model.set_env(env)
        if pretrain:
            env_expert = DummyVecEnv([lambda: ControllerEnv(manual_control=False)])
            self.model = pretrain_agent(self.model, env_expert, num_interactions=50e4)
        cb1 = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        cb2 = EvalCallback(self.model.env, eval_freq=5000, best_model_save_path=os.path.join(self.log_dir, 'eval'))
        cb3 = EarlyStopping(10000, 10, self.log_dir, verbose=1)
        cb = CallbackList([cb1, cb3]) #, cb2])
        self.model.learn(total_timesteps=timesteps, callback=cb)

    def test_env(self, num_interactions:int, env, no_action=False, use_render=False):
        if not no_action:
            self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name))
        done = False
        obs = env.reset()
        data = {}
        state = None
        for i in tqdm(range(num_interactions), desc="Тестирование модели"):
            if no_action:
                action = [0] if env.envs[0].env.ctrl.manual_control else [1]
            else:
                action, state = self.model.predict(obs, state=state, deterministic=True)
            obs, reward, done, data = env.step(action)
            if use_render:
                env.render()
        return data

    def test(self, *ctrl_env_args, ht_func=lambda t: 11000, **ctrl_env_kwargs):
        self.env = ControllerEnv(*ctrl_env_args, h_func=ht_func, use_storage=True, is_testing=True, **ctrl_env_kwargs)
        tk = ctrl_env_kwargs['tk']
        print('Расчет перехода с использованием нейросетевого регулятора [ht_func]')
        env = self._wrap_env(self.env)
        self.test_env(int(tk/self.env.ctrl.model.dt)-1, env, use_render=True)
        storage1 = env.envs[0].env.ctrl.storage
        y_t = [ht_func(t) for t in storage1.storage['t']]
        storage1.storage['y_real'] = y_t
        y_neural, varth_neural = storage1.storage["y"], storage1.storage["vartheta"]
        storage1.plot(["deltaz"], "t")
        storage1.plot(["vartheta", 'vartheta_ref'], "t")
        storage1.plot(["y", 'y_real'], "t")
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=5)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")

        print('Расчет перехода с использованием ПИД-регулятора [ht_func]')
        env = ControllerEnv(h_func=ht_func, manual_control=False, use_storage=True, is_testing=True, tk=tk)
        env = self._wrap_env(env)
        self.test_env(int(tk/self.env.ctrl.model.dt)-1, env, no_action=True, use_render=True)
        storage2 = env.envs[0].env.ctrl.storage
        storage2.plot(["vartheta_neural", "vartheta"], "t")
        storage2.plot(["y_neural", "y"], "t")
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=15)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")


if __name__ == '__main__':
    net_class = A2C
    manual_control = True
    n_actions = None
    # ============ Обучение ============
    tk_train = 60 # секунд
    pretrain = True
    optimize = True
    train = True
    # ========== Тестирование ==========
    tk_test = 50 # секунд
    ht_test = lambda t: 12000
    # ==================================
    ctrl = ControllerAgent(net_class=net_class, use_tb=True)
    if train:
        ctrl.train(timesteps=100000, pretrain=pretrain, optimize=optimize, n_actions=n_actions, tk=tk_train, manual_control=manual_control)
    ctrl.test(ht_func=ht_test, n_actions=n_actions, tk=tk_test, manual_control=manual_control)