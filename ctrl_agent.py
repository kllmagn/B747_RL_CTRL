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
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from callbacks import *
from tensorboard import program

from typing import Callable

from tqdm import tqdm

import torch as th

import optuna
from scipy import optimize

import onnx
import onnxruntime as ort

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


class ControllerAgent:

    def __init__(self, net_class=A2C, use_tb=False, log_dir='./logs'):
        th.manual_seed(1) # выставляем seed для детерминированного поведения
        self.log_dir = log_dir # папка с логами
        self.tb_log = os.path.join(self.log_dir, 'tb_log') if use_tb else None # ссылка на папку с логами обучения
        if use_tb: # если используем TensorBoard, то запускаем его
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_log])
            self.tb_url = self.tb.launch()
            print(f"TensorBoard listening on {self.tb_url}")
        else: # если не используем, то выставляем в None
            self.tb, self.tb_url = None, None
        self.net_class = net_class # сохраняем используемый класс нейросети
        if self.net_class in hyperparams: # если класс присутствует в базе гиперпараметров, то получаем соответствующие гиперпараметры
            print('Using existing model configuration.')
            self.hp = hyperparams[self.net_class]
        else: # если класса нет, то используются гиперпараметры по умолчанию
            self.hp = {}
        self.model = None # выставляем объект модели нейросети как пустой
        self.bm_name = 'best_model.zip' # выставляем имя файла модели


    def _wrap_env(self, env, monitor_dir=None, manual_reset=False, use_monitor=True):
        '''
        Функция для обертки среды соответствующими классами.
        '''
        if use_monitor:
            env = Monitor(env, os.path.join((monitor_dir if monitor_dir else self.log_dir), 'monitor.csv'))
        env = DummyVecEnv([lambda: env], manual_reset=manual_reset)
        #env = VecNormalize(env, gamma=0.95, norm_obs = False, norm_reward=True)
        env.seed(1)
        return env


    def _unwrap_env(self, env):
        '''
        Функция обратной обертки среды.
        '''
        return env.envs[0].env


    def optimize(self, training_timesteps, *ctrl_env_args, pretrain=False, opt_max=True, opt_hp=True, **ctrl_env_kwargs):
        '''
        Оптимизировать нейросетевую модель.
        '''
        def save_model_callback(study:optuna.Study, trial):
            if (opt_max and study.best_value <= trial.value) or (not opt_max and study.best_value >= trial.value):
                load_path = os.path.join(self.log_dir, 'best_model.zip')
                save_path = os.path.join(self.log_dir, 'optimization', 'best_model.zip')
                os.replace(save_path, load_path)
        def objective(trial:optuna.Trial):
            if opt_hp:
                if self.net_class is A2C:
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
            else:
                hp = self.hp
            reward_config = {}
            '''
                'k1': trial.suggest_float('k1', 0.1, 1),
                'k2': trial.suggest_float('k2', 0.1, 1),
                'k3': trial.suggest_float('k3', 0.1, 1)
                '''
            env = ControllerEnv(*ctrl_env_args, reward_config=reward_config, use_storage=True, **ctrl_env_kwargs)
            env = self._wrap_env(env, os.path.join(self.log_dir, 'optimization'))
            self.model = self.net_class('MlpPolicy', env, verbose=0, tensorboard_log=self.tb_log, **hp)
            if pretrain:
                env_expert = DummyVecEnv([lambda: ControllerEnv(use_ctrl=ctrl_env_kwargs['use_ctrl'], no_correct=True)])
                self.model = pretrain_agent_imit(self.model, env_expert, timesteps=50000, num_episodes=50)
                es_startup = 20000
            else:
                es_startup = 0
            total_timesteps = training_timesteps
            savebest_dir = os.path.join(self.log_dir, 'optimization')
            cb2 = SaveOnBestTrainingRewardCallback(10000, savebest_dir, 1)
            #cb2 = SaveOnBestQualityMetricCallback(lambda env: env.get_attr('ctrl')[0].model.TAE, 'TAE', 10000, log_dir=savebest_dir, maximize=opt_max) #vth_err_abs.output()
            cb = EarlyStopping(lambda: cb2.mean_reward, 'mean_reward', 10000, 2, verbose=1, startup_step=es_startup, maximize=opt_max)
            with ProgressBarManager(total_timesteps) as callback:
                cb_list = CallbackList([callback, cb2, cb]) #, cb, cb2])
                self.model.learn(total_timesteps=total_timesteps, callback=cb_list)
            self.model = self.net_class.load(os.path.join(savebest_dir, 'best_model.zip'))
            return cb2.best_mean_reward

        study = optuna.create_study(direction=("maximize" if opt_max else "minimize"))
        study.optimize(objective, n_trials=500, callbacks=[save_model_callback])
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
            self.model = self.net_class.load(preload_path, tensorboard_log=self.tb_log)
            self.model.set_env(env)
        else:
            self.model = self.net_class('MlpPolicy', env, tensorboard_log=self.tb_log, **self.hp)
        env_expert = DummyVecEnv([lambda: ControllerEnv(use_ctrl=ctrl_env_kwargs['use_ctrl'], no_correct=True, manual_ctrl=False, manual_stab=False)])
        self.model = pretrain_agent_imit(self.model, env_expert, timesteps=timesteps, num_episodes=num_int_episodes, algo=algo)
        self.model.save(os.path.join(self.log_dir, self.bm_name))


    def train(self, *ctrl_env_args, timesteps=50000, preload=False, use_es=True, optimize=False, opt_max=True, opt_hp=True, verbose:int=1, log_interval:int=1000, **ctrl_env_kwargs):
        '''
        Произвести обучение нейросетевой модели.
        '''
        self.env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(self.env)
        if optimize:
            print('Оптимизирую с помощью Optuna')
            self.optimize(timesteps, opt_max=opt_max, opt_hp=opt_hp, *ctrl_env_args, **ctrl_env_kwargs)
            self.model = self.net_class.load(os.path.join(self.log_dir, 'best_model.zip'), tensorboard_log=self.tb_log, verbose=verbose)
            self.model.set_env(env)
        else:
            if preload:
                preload_path = os.path.join(self.log_dir, 'best_model.zip')
                print('Предзагружаю модель из', preload_path)
                self.model = self.net_class.load(preload_path, tensorboard_log=self.tb_log, verbose=verbose)
                self.model.set_env(env)
            else:
                print('Создаю новую модель:', str(self.net_class))
                self.model = self.net_class('MlpPolicy', env, tensorboard_log=self.tb_log, verbose=verbose, **self.hp)
        cb1 = SaveOnBestTrainingRewardCallback(10000, self.log_dir, 1)
        cb_metric = SaveOnBestQualityMetricCallback(lambda env: env.get_attr('ctrl')[0].deltaz_diff_int.output(), 'deltaz_diff_int', 10000, log_dir=self.log_dir, maximize=False)
        cb2 = EarlyStopping(lambda: cb1.mean_reward, 'vth_err', 10000, 4, verbose=1, maximize=True)
        cbs = [cb1]
        if use_es:
            cbs.append(cb2)
        cb = CallbackList(cbs)
        self.model.learn(total_timesteps=timesteps, callback=cb, log_interval=log_interval)


    def convert_to_onnx(self, filename:str):
        '''
        Произвести конвертацию модели в обобщенный формат .onnx.
        '''
        self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name))
        if type(self.model) not in [PPO, A2C]:
            raise NotImplementedError
        class OnnxablePolicy(th.nn.Module):
            def __init__(self, extractor, action_net, value_net):
                super(OnnxablePolicy, self).__init__()
                self.extractor = extractor
                self.action_net = action_net
                self.value_net = value_net
            def forward(self, observation):
                # NOTE: You may have to process (normalize) observation in the correct
                #       way before using this. See `common.preprocessing.preprocess_obs`
                action_hidden, value_hidden = self.extractor(observation)
                #print(action_hidden, value_hidden)
                return self.action_net(action_hidden), self.value_net(value_hidden)
        self.model.policy.to("cpu")
        onnxable_model = OnnxablePolicy(self.model.policy.mlp_extractor, self.model.policy.action_net, self.model.policy.value_net)
        dummy_input = th.randn(1, 3)
        th.onnx.export(onnxable_model, dummy_input, filename, opset_version=11, verbose=True)


    def test_onnx(self, filename:str):
        '''
        Протестировать существующий файл модели формата .onnx.
        '''
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        observation = np.zeros((1, 3)).astype(np.float32)
        ort_sess = ort.InferenceSession(filename)
        action, value = ort_sess.run(None, {'input.1': observation})
        action = np.clip(action, -17*pi/180, 17*pi/180)
        print('action:', action, 'value:', value)


    def test_env(self, num_interactions:int, env, no_action=False, use_render=False, on_episode_end=None):
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
            obs, _, done, _ = env.step(action)
            if done:
                if on_episode_end:
                    on_episode_end(env)
                storage = copy.deepcopy(tmp_storage)
                rews.append(env.buf_infos[0]['episode']['r'])
                obs = env.reset()
            else:
                tmp_storage = env.get_attr('ctrl')[0].storage
            if use_render:
                env.render()
        return np.mean(rews), np.std(rews), storage


    def test(self, *ctrl_env_args, ht_func=None, varthetat_func=None, **ctrl_env_kwargs):
        ctrl_env_kwargs['random_reset'] = False

        self.env = ControllerEnv(*ctrl_env_args, h_func=ht_func, vartheta_func=varthetat_func, use_storage=True,\
            is_testing=True, **ctrl_env_kwargs)

        tk = ctrl_env_kwargs['tk']
        print('Расчет перехода с использованием нейросетевого регулятора [func]')
        env = self._wrap_env(self.env, manual_reset=True)
        hf = vf = None
        def callb(env):
            nonlocal vf, hf
            vf = env.get_attr('ctrl')[0].vartheta_func
            hf = env.get_attr('ctrl')[0].h_func
            ctrl_obj = env.get_attr('ctrl')[0]
            if ctrl_obj.use_ctrl:
                print(ctrl_obj.stepinfo_CS())
            else:
                print(ctrl_obj.stepinfo_SS())
            print('Суммарная ошибка по углу:', ctrl_obj.vth_err.output())
        num_interactions = int(tk/self.env.ctrl.sample_time)
        mean_reward, std_reward, storage1 = self.test_env(num_interactions, env, use_render=True, on_episode_end=callb)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")

        print('Расчет перехода с использованием ПИД-регулятора [func]')
        env = ControllerEnv(h_func=hf, vartheta_func=vf, use_ctrl=ctrl_env_kwargs['use_ctrl'],\
            no_correct=True, manual_ctrl=False, manual_stab=False, use_storage=True, is_testing=True, tk=tk, random_reset=ctrl_env_kwargs['random_reset'])
        env = self._wrap_env(env, manual_reset=True)
        mean_reward, std_reward, storage2 = self.test_env(\
            num_interactions*int(self.env.ctrl.sample_time/self.env.ctrl.model.dt),\
                env, no_action=True, use_render=True, on_episode_end=callb)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")

        storage2.merge(storage1, 'neural')
        storage2.plot(["vartheta_ref", "vartheta_ref_neural", "vartheta_neural", "vartheta"], "t", 't, [с]', 'ϑ, [град]')
        if env.envs[0].env.ctrl.use_ctrl:
            storage2.plot(['hzh', 'hzh_neural', 'y_neural', 'y'], "t", 't, [с]', 'h, [м]')
        storage2.plot(['deltaz_neural', 'deltaz', 'deltaz_ref_neural'], 't', 't, [с]', 'δ_ком, [град]')
        storage2.plot(["deltaz_real_neural", "deltaz_real"], "t", 't, [с]', 'δ, [град]')

        return storage2


    def show(self):
        '''
        Показать структуру модели.
        '''
        if self.model is not None:
            attrs = ['gamma', 'max_grad_norm', 'gae_lambda', 'n_steps', 'learning_rate', 'ent_coef', 'vf_coef']
            print(self.model.policy.optimizer_class)
            for k in attrs:
                print(k, ':', self.model.__dict__[k])
            print('='*20)
            print(self.model.policy)
        else:
            print('Невозможно отобразить структуру модели: модель отсутствует.')


if __name__ == '__main__':
    # добавить таймер для обучения
    net_class = A2C
    use_tb = True
    log_interval = 1000
    env_kwargs = dict(
        use_ctrl = False, # использовать СУ (ПИД-регулятор авто или коррекция)
        manual_ctrl = False, # вкл. ручное управление СУ (откл. поддержку ПИД-регулятора)
        manual_stab = True, # вкл. ручное управление СС (откл. поддержку ПИД-регулятора)
        no_correct = True, # не использовать коррекцию коэффициентов ПИД-регуляторов
        sample_time = 0.05,
        use_limiter = False,
        random_init = True, # случайная инициализация начального состояния
        #reward_config={'k1': 0.5626263389608758, 'k2': 0.957988620443826, 'k3': 0.20433884176957848} #{'kv': 23.02559907773439, 'kw': 123.40541803849644, 'kdeltaz': 6.523852550774975}
    )
    # ===== Имитационное обучение ======
    pretrain = False 
    pretrain_kwargs = dict(
        timesteps = 1_000_000, # epochs (BC)
        preload = False,
        num_int_episodes = 200,
        algo = 'GAIL' # BC, GAIL, AIRL
    )
    # ============ Обучение =============
    train = True
    train_kwargs = dict(
        timesteps = 500000,
        tk = 10, # секунд
        preload = False,
        use_es = False,
        optimize = False,
        opt_max = True,
        opt_hp = True,
        verbose=int(use_tb),
        log_interval=log_interval
    )
    # ========== Тестирование ==========
    test_kwargs = dict(
        tk = 60, # секунд
        ht_func = lambda t: 11500,
        varthetat_func = lambda t: 10*pi/180 #if t < 25 else -10*pi/180
    )
    # ==================================
    ctrl = ControllerAgent(net_class=net_class, use_tb=use_tb)
    if pretrain:
        ctrl.pretrain(**pretrain_kwargs, **env_kwargs)
    if train:
        ctrl.train(**train_kwargs, **env_kwargs)
    #ctrl.test(**test_kwargs, **env_kwargs)
    # ==================================
    varthetas = [5*pi/180] #, 10*pi/180, -10*pi/180, -5*pi/180]
    hs = [] #10000, 10500, 11500, 12000]
    for i in range(len(varthetas)):
        print('='*30)
        print('Тестирую угол тангажа vartheta =', varthetas[i]*180/pi, '[град]')
        env_kwargs['use_ctrl'] = False
        storage = ctrl.test(tk=test_kwargs['tk'], ht_func = lambda t: 11000, varthetat_func = lambda t: varthetas[i], **env_kwargs)
        storage.save(f'data_vartheta_{varthetas[i]*180/pi}.xlsx')
    for i in range(len(hs)):
        print('='*30)
        print('Тестирую высоту h =', hs[i], '[м]')
        env_kwargs['use_ctrl'] = True
        storage = ctrl.test(tk=test_kwargs['tk'], ht_func = lambda t: hs[i], varthetat_func = lambda t: 10*pi/180, **env_kwargs)
        storage.save(f'data_h_{hs[i]}.xlsx')
    ctrl.show()
    ctrl.convert_to_onnx('model.onnx')
    ctrl.test_onnx('model.onnx')