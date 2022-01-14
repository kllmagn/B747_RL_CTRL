import os
import copy
from math import pi
import random

import numpy as np

from nirs.envs.ctrl_env.ctrl_env import ControllerEnv
from pretrain import pretrain_agent, pretrain_agent_imit

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
    SAC: {
        #'tau': 0.995,
        'buffer_size': int(5e5),
        #'learning_rate': linear_schedule(4e-4),
        #'policy_kwargs': dict(net_arch=[64, 64]),
        #'target_entropy': -3,
        'learning_starts': int(10000)
    },
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
    }
}

# Trial 25 finished with value: -61.18436341101212 and parameters: {'use_rms_prop': False, 'learning_rate': 0.00030937810859534454, 'n_depth': 3, 'n1': 329, 'n2': 316, 'n3': 370}. 

class ControllerAgent:
    def __init__(self, net_class=A2C, use_tb=False, log_dir='./logs'):
        th.manual_seed(1)
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

    def _wrap_env(self, env, monitor_dir=None, manual_reset=False, use_monitor=True):
        if use_monitor:
            env = Monitor(env, os.path.join((monitor_dir if monitor_dir else self.log_dir), 'monitor.csv'))
        env = DummyVecEnv([lambda: env], manual_reset=manual_reset)
        #env = VecNormalize(env, gamma=0.95, norm_reward=False)
        #env.seed(1)
        return env

    def _unwrap_env(self, env):
        return env.envs[0].env

    def optimize(self, training_timesteps, *ctrl_env_args, pretrain=False, pretrain_ias=50e3, **ctrl_env_kwargs):
        def save_model_callback(study:optuna.Study, trial):
            if study.best_value <= trial.value:
                load_path = os.path.join(self.log_dir, 'best_model.zip')
                save_path = os.path.join(self.log_dir, 'optimization', 'best_model.zip')
                os.replace(save_path, load_path)
                #print(f"Getting best trial from {load_path}")
                #self.model.load(load_path)
                #env = self.model.env
                #elf.test_env(5000-1, env)
                #env.envs[0].env.ctrl.storage.save(base='t')
                #env.envs[0].env.ctrl.storage.plot(['vartheta', 'vartheta_ref'], 't')
        def objective(trial:optuna.Trial):
            if self.net_class is A2C:
                hp = {
                #'gamma': trial.suggest_float('gamma', 0.7, 0.99),
                #'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 0.8),
                'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
                #'gae_lambda': trial.suggest_float('gae_lambda', 0.6, 1.0),
                #'n_steps': trial.suggest_int('n_steps', 1, 256),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
                'ent_coef': 0.0026650043954570186,
                'vf_coef': 0.10796014008883446,
                #'ent_coef': trial.suggest_float('ent_coef', 0, 0.01),
                #'vf_coef': trial.suggest_float('vf_coef', 0.05, 0.6),
                'policy_kwargs': dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[trial.suggest_int(f'n{i+1}', 300, 400) for i in range(trial.suggest_int('n_depth', 2, 3))]) #dict(pi=[254,254], vf=[254,254])])
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
            elif self.net_class is SAC:
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
                hp = self.hp
            env = ControllerEnv(*ctrl_env_args, use_storage=True, **ctrl_env_kwargs)
            env = self._wrap_env(env, os.path.join(self.log_dir, 'optimization'))
            self.model = self.net_class('MlpPolicy', env, verbose=0, tensorboard_log=self.tb_log, **hp)
            if pretrain:
                env_expert = DummyVecEnv([lambda: ControllerEnv(use_ctrl=ctrl_env_kwargs['use_ctrl'], full_auto=True)])
                self.model = pretrain_agent_imit(self.model, env_expert, timesteps=50000, num_episodes=50)
                es_startup = 20000
            else:
                es_startup = 0
            total_timesteps = training_timesteps
            savebest_dir = os.path.join(self.log_dir, 'optimization')
            cb = EarlyStopping(10000, 4, savebest_dir, verbose=1, startup_step=es_startup)
            cb1 = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=savebest_dir)
            with ProgressBarManager(total_timesteps) as callback:
                cb_list = CallbackList([callback, cb1, cb])
                self.model.learn(total_timesteps=total_timesteps, callback=cb_list)
            self.model = self.net_class.load(os.path.join(savebest_dir, 'best_model.zip'))
            #return cb1.best_mean_reward
            mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=10, render=True)
            print(f"Mean reward = {mean_reward} +/- {std_reward}")
            return mean_reward

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, callbacks=[save_model_callback])
        params = study.best_params
        print('Лучшие параметры:', params)
        params['policy_kwargs'] = dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[params[f'n{i+1}'] for i in range(params['n_depth'])])
        for i in range(params['n_depth']):
            del params[f'n{i+1}'] 
        del params['n_depth']
        self.hp = params
        print('Параметры нейросети:', self.hp)

    def pretrain(self, *ctrl_env_args, timesteps:int=50000, preload=False, num_int_episodes:int=100, algo:str='BC', **ctrl_env_kwargs):
        self.env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(self.env)
        if preload:
            preload_path = os.path.join(self.log_dir, self.bm_name)
            print('Предзагружаю модель из', preload_path)
            self.model = self.net_class.load(preload_path)
            self.model.set_env(env)
        else:
            self.model = self.net_class('MlpPolicy', env, **self.hp)
        env_expert = DummyVecEnv([lambda: ControllerEnv(use_ctrl=ctrl_env_kwargs['use_ctrl'], full_auto=True)])
        self.model = pretrain_agent_imit(self.model, env_expert, timesteps=timesteps, num_episodes=num_int_episodes, algo=algo)
        self.model.save(os.path.join(self.log_dir, self.bm_name))

    def train(self, *ctrl_env_args, timesteps=50000, preload=False, optimize=False, **ctrl_env_kwargs):
        self.env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(self.env)
        if optimize:
            print('Оптимизирую с помощью Optuna')
            self.optimize(timesteps, *ctrl_env_args, **ctrl_env_kwargs)
            self.model = self.net_class.load(os.path.join(self.log_dir, 'best_model.zip'))
            self.model.set_env(env)
        else:
            if preload:
                preload_path = os.path.join(self.log_dir, 'best_model.zip')
                print('Предзагружаю модель из', preload_path)
                self.model = self.net_class.load(preload_path)
                self.model.set_env(env)
            else:
                self.model = self.net_class('MlpPolicy', env, **self.hp)
        cb1 = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        cb2 = EvalCallback(self.model.env, eval_freq=5000, best_model_save_path=os.path.join(self.log_dir, 'eval'))
        cb3 = EarlyStopping(10000, 20, self.log_dir, verbose=1)
        cb = CallbackList([cb1, cb3]) #, cb2])
        self.model.learn(total_timesteps=timesteps, callback=cb)

    def test_env(self, num_interactions:int, env, no_action=False, use_render=False):
        if not no_action:
            self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name))
        done = False
        obs = env.reset()
        state = None
        storage = None
        tmp_storage = None
        rews = []
        for i in tqdm(range(num_interactions), desc="Тестирование модели"):
            if no_action:
                action = [0] if env.envs[0].env.ctrl.manual_stab else [env.envs[0].env.ctrl.model.deltaz_ref]
            else:
                action, state = self.model.predict(obs, state=state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if done:
                storage = copy.deepcopy(tmp_storage)
                rews.append(env.buf_infos[0]['episode']['r'])
                obs = env.reset()
            else:
                tmp_storage = env.get_attr('ctrl')[0].storage
            if use_render:
                env.render()
        return np.mean(rews), np.std(rews), storage

    def test(self, *ctrl_env_args, ht_func=None, varthetat_func=None, **ctrl_env_kwargs):
        if ht_func is None:
            ht = random.uniform(10800, 11300)
            ht_func = lambda t: ht
        if varthetat_func is None:
            varthetat= random.uniform(-10*pi/180, 10*pi/180)
            varthetat_func = lambda t: varthetat
        self.env = ControllerEnv(*ctrl_env_args, h_func=ht_func, vartheta_func=varthetat_func, use_storage=True,\
            is_testing=True, **ctrl_env_kwargs)
        tk = ctrl_env_kwargs['tk']
        print('Расчет перехода с использованием нейросетевого регулятора [func]')
        env = self._wrap_env(self.env, manual_reset=True)
        num_interactions = int(tk/self.env.ctrl.sample_time)
        mean_reward, std_reward, storage1 = self.test_env(num_interactions, env, use_render=True)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")
        y_neural, varth_neural = storage1.storage["y"], storage1.storage["vartheta"]
        storage1.plot(["deltaz"], "t")
        storage1.plot(["vartheta", 'vartheta_ref'], "t")
        if env.envs[0].env.ctrl.use_ctrl:
            storage1.plot(["y", 'hzh'], "t")
        #mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=5)
        #print(f"Mean reward = {mean_reward} +/- {std_reward}")

        print('Расчет перехода с использованием ПИД-регулятора [func]')
        env = ControllerEnv(h_func=ht_func, vartheta_func=varthetat_func, use_ctrl=ctrl_env_kwargs['use_ctrl'], full_auto=True, use_storage=True, is_testing=True, tk=tk)
        env = self._wrap_env(env, manual_reset=True)
        mean_reward, std_reward, storage2 = self.test_env(num_interactions*int(self.env.ctrl.sample_time/self.env.ctrl.model.dt), env, no_action=True, use_render=True)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")
        storage2.merge(storage1, 'neural')
        storage2.plot(["deltaz_neural", "deltaz"], "t")
        storage2.plot(["vartheta_ref", "vartheta_neural", "vartheta"], "t")
        if env.envs[0].env.ctrl.use_ctrl:
            storage1.plot(["y", 'hzh'], "t")
            storage2.plot(["y_neural", "y"], "t")


if __name__ == '__main__':
    net_class = A2C
    env_kwargs = dict(
        use_ctrl = False, # использовать СУ (ПИД-регулятор или нейросетевое обеспечение)
        manual_ctrl = False, # вкл. ручное управление СУ (откл. поддержку ПИД-регулятора)
        manual_stab = True, # вкл. ручное управление СС (откл. поддержку ПИД-регулятора)
        n_actions = None,
        sample_time = 0.05,
        use_limiter = False
    )
    # ===== Имитационное обучение ======
    pretrain = False
    pretrain_kwargs = dict(
        timesteps = 1_000_000, # epochs (BC)
        preload = False,
        num_int_episodes = 200,
        algo = 'GAIL' # BC, GAIL, AIRL
    )
    # ============ Обучение ============
    train = False
    train_kwargs = dict(
        timesteps =  1_000_000,
        tk = 80, # секунд
        preload = False,
        optimize = True
    )
    # ========== Тестирование ==========
    test_kwargs = dict(
        tk = 50, # секунд
        ht_func = lambda t: 12000,
        varthetat_func = lambda t: 10*pi/180
    )
    # ==================================
    ctrl = ControllerAgent(net_class=net_class, use_tb=True)
    if pretrain:
        ctrl.pretrain(**pretrain_kwargs, **env_kwargs)
    if train:
        ctrl.train(**train_kwargs, **env_kwargs)
    ctrl.test(**test_kwargs, **env_kwargs)