import os
import copy
from math import pi
from pathlib import Path
import random
from threading import Thread
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from core.controller import CtrlType

from tools.general import Storage, get_model_name_desc

from .callbacks import *
from tensorboard import program

from tqdm import tqdm

import optuna

import onnx
import onnxruntime as ort

from env.ctrl_env import *
from .setups import hyperparams, trial_hyperparams, TrainPlotter


class ControllerAgent:
    '''Агент для взамодействия со средой.'''

    def __init__(self, net_class=A2C, use_tb=False, log_dir='./.logs', model_name='best_model'):
        '''Инициализировать объект агента.'''
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
            hyps = hyperparams[self.net_class]
            print(f'Использую существующую конфигурацию модели: {hyps}')
            self.hp = hyps
        else: # если класса нет, то используются гиперпараметры по умолчанию
            self.hp = {}
        self.model = None # выставляем объект модели нейросети как пустой
        self._init_names(model_name)


    def _init_names(self, model_name:str) -> None:
        self.model_name = model_name
        self.bm_name = f'{model_name}.zip' # выставляем имя файла модели


    def _wrap_env(self, env_init:Callable[[], ControllerEnv], monitor_dir=None, manual_reset=False, use_monitor=True) -> DummyVecEnv:
        '''Функция для обертки среды соответствующими классами.'''
        n_cpu = 4
        monitor_path = os.path.join((monitor_dir if monitor_dir else self.log_dir), 'monitor.csv')
        def _env_init() -> ControllerEnv:
            env = env_init()
            if n_cpu == 1 and use_monitor:
                env = Monitor(env, monitor_path)
            return env
        envs = [_env_init for i in range(n_cpu)]
        if n_cpu > 1:
            env = SubprocVecEnv(envs)
        else:
            env = DummyVecEnv(envs, manual_reset=manual_reset)
        if use_monitor:
            env = VecMonitor(env, monitor_path)
        #env = VecNormalize(env, gamma=0.95, norm_obs = False, norm_reward=True)
        env.seed(1)
        return env


    def _unwrap_env(self, env:DummyVecEnv) -> ControllerEnv:
        '''Функция обратной обертки среды.'''
        return env.envs[0].env


    def optimize(self, env_init_func:Callable[[],ControllerEnv], training_timesteps:int, opt_hp=False, verbose:int=1) -> None:
        '''Оптимизировать нейросетевую модель.'''
        savebest_dir = os.path.join(self.log_dir, 'optimization') 
        def save_model_callback(study:optuna.Study, trial):
            '''Функция для сохранения в файл модели наилучшей итерации процесса оптимизации.'''
            if study.best_value >= trial.value:
                load_path = os.path.join(self.log_dir, self.bm_name)
                save_path = os.path.join(savebest_dir, self.bm_name)
                os.replace(save_path, load_path)
        def objective(trial:optuna.Trial):
            if opt_hp:
                hp = trial_hyperparams(self.net_class, trial, self.hp) # получем гиперпараметры сети для данной итерации
            else:
                hp = self.hp
            env = env_init_func()
            reward_config = get_rew_config(env.reward_type, trial)
            del env
            def env_init_func_patched() -> ControllerEnv:
                env = env_init_func()
                env.set_rew_config(reward_config)
                return env
            env = self._wrap_env(env_init_func_patched, os.path.join(self.log_dir, 'optimization')) # оборачиваем среду в векторное представление
            if opt_hp or self.model is None:
                self.model = self.net_class('MlpPolicy', env, verbose=0, **hp) # создаем модель
            else:
                self.model.set_env(env)
            control_test = ControlTestCallback(
                    net_class=self.net_class,
                    env_gen=env_init_func_patched,
                    vartheta_ref=[5*pi/180, -5*pi/180, 10*pi/180, -10*pi/180],
                    state0=np.array([0, 11000, 250, 0, 0, 0]),
                    log_interval=1000,
                    filename=self.bm_name,
                    log_dir=savebest_dir,
                    window_length=30,
                    verbose=verbose,
                )
            if verbose > 0:
                with ProgressBarManager(training_timesteps) as callback:
                    cb_list = CallbackList([callback, control_test])
                    self.model.learn(total_timesteps=training_timesteps, callback=cb_list)
            else:
                self.model.learn(total_timesteps=training_timesteps, callback=CallbackList([control_test]))
            self.model = self.net_class.load(os.path.join(savebest_dir, self.bm_name))
            return control_test.mean_quality

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=500, callbacks=[save_model_callback], catch=(ValueError,))
        print('Лучшие параметры:', study.best_params)
        '''
        params = dict(study.best_params)
        params['policy_kwargs'] = dict(ortho_init=False, activation_fn=th.nn.Tanh, net_arch=[params[f'n{i+1}'] for i in range(params['n_depth'])])
        for i in range(params['n_depth']):
            del params[f'n{i+1}'] 
        del params['n_depth']
        self.hp = params
        print('Полученные гиперпараметры нейросетевой модели:', self.hp)
        '''


    def train(
            self,
            env_init_func:Callable[[], ControllerEnv],
            timesteps=50000,
            preload:Union[bool, str]=False,
            optimize=False,
            verbose:int=1,
            log_interval:int=1000,
            show_plotter:bool=False,
            callbacks_init:List[Callable[[ControllerEnv], Any]]=[],
            reward_config:dict={},
        ):
        '''Произвести обучение нейросетевой модели.'''
        def env_init_func_patched() -> ControllerEnv:
            env = env_init_func()
            env.set_rew_config(reward_config)
            return env
        env = self._wrap_env(env_init_func_patched) # оборачиваем объект в векторное представление
        if optimize: # если производится предварительная оптимизация
            print('Оптимизирую с помощью Optuna')
            self.optimize(env_init_func, timesteps, verbose=verbose) 
            self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name), tensorboard_log=self.tb_log, verbose=verbose)
            self.model.set_env(env)
        else:
            if preload: # если модель предзагружается из файла
                if type(preload) is str:
                    preload_path = preload
                else:
                    preload_path = os.path.join(self.log_dir, self.bm_name)
                print('Предзагружаю модель из', preload_path)
                self.model = self.net_class.load(preload_path, tensorboard_log=self.tb_log, verbose=verbose)
                self.model.set_env(env)
            else:
                print('Создаю новую модель:', str(self.net_class))
                self.model = self.net_class('MlpPolicy', env, tensorboard_log=self.tb_log, verbose=verbose, **self.hp)
        cb = CallbackList([cb_init(self) for cb_init in callbacks_init])
        tplotter = None
        if show_plotter:
            def tplotter_gen(env):
                nonlocal tplotter
                tplotter = TrainPlotter(env, ['vartheta', 'vartheta_ref'], 't')
                if show_plotter:
                    tplotter.show()
            t = Thread(target=tplotter_gen, args=(env,))
            t.start()
        self.model.learn(total_timesteps=timesteps, callback=cb, log_interval=log_interval, tb_log_name=self.model_name)


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


    def _test_env(self, ref_value:float, num_interactions:int, env:ControllerEnv, state0:np.ndarray, manual=True, use_render=False, on_episode_end=None) -> None:
        if manual:
            self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name))
        done = False

        rew = None
        def _post_step(self:Controller, state=None):
            nonlocal rew
            env.ctrl.storage.record('rew', rew)
            Controller._post_step(self, state)
        env.ctrl._post_step = _post_step.__get__(env.ctrl, Controller)
        
        obs = env.reset(state0)
        if env.ctrl.use_ctrl:
            env.ctrl.h_func = lambda _: ref_value
        else:
            env.ctrl.vartheta_func = lambda _: ref_value
        state = None
        for _ in tqdm(range(num_interactions), desc="Тестирование модели"):
            if manual:
                action, state = self.model.predict(obs, state=state, deterministic=True)
            else:
                action = None
            obs, _, done, _ = env.step(action)
            rew = env.get_reward(action)
            if done:
                if on_episode_end:
                    on_episode_end(env)
                #obs = env.reset()
            if use_render:
                env.render()


    def test(
        self,
        ref_values:List[float],
        env_init_func:Union[Dict[str, Callable[[], ControllerEnv]], Callable[[], ControllerEnv]],
        state0:np.ndarray=None,
        plot=False,
        output_dir:Path=None,
        collect=False,
        no_neural=False,
        pid_coefs:List[np.ndarray]=[],
        ) -> None:
               
        info = None
        tmp_storage = None
        quality = None

        def callb(env:ControllerEnv) -> None:
            nonlocal tmp_storage
            nonlocal info
            nonlocal quality
            tmp_storage = env.ctrl.storage
            ctrl_obj = env.ctrl
            if ctrl_obj.use_ctrl:
                info = ctrl_obj.stepinfo_CS()
            else:
                info = ctrl_obj.stepinfo_SS()
            quality = ctrl_obj.quality()

        def set_pid_coefs(env:ControllerEnv, coefs:np.ndarray) -> ControllerEnv:
            if env.ctrl.use_ctrl:
                env.ctrl.model.PID_CS = coefs
            else:
                env.ctrl.model.PID_SS = coefs
            return env

        env_inits = env_init_func if type(env_init_func) is dict else {self.model_name: env_init_func}

        env_PID = env_inits[list(env_inits.keys())[0]]()
        env_PID.ctrl.ctrl_type = CtrlType.AUTO if (env_PID.ctrl.ctrl_type == CtrlType.MANUAL) else CtrlType.FULL_AUTO
        env_PID.ctrl.ctrl_mode = None
        env_PID.ctrl.use_storage = True
        env_PID.ctrl.sample_time = env_PID.ctrl.model.dt
        del env_PID.ctrl.model
        env_PID.ctrl._init_model()
        tk = env_PID.ctrl.tk

        model_name_backup = self.model_name
        model_names = [] if no_neural else list(env_inits.keys())

        datas = []
        storages = {}

        base_pid_name = "CУ ПИД" if env_PID.ctrl.use_ctrl else "СС ПИД"
        def pid_name_index(i:int) -> str:
            pid_index = f" [{i+1}]" if len(pid_coefs) > 1 else ""
            name = f"{base_pid_name}{pid_index}"
            return name

        if len(pid_coefs) == 0:
            pid_coefs = [env_PID.ctrl.model.PID_CS if env_PID.ctrl.use_ctrl else env_PID.ctrl.model.PID_SS]

        for ref_value in ref_values:
            print('='*60)
            storage = None
            
            data = pd.DataFrame(columns=['Устройство', 'σ, [%]', 'tпп, [с]', 'tв, [с]', f"Δ, {'[м]' if env_PID.ctrl.use_ctrl else '[град]'}", 'Q, [-]'])

            for i in range(len(pid_coefs)):
                coefs = pid_coefs[i]

                env_PID = set_pid_coefs(env_PID, coefs)
                name = pid_name_index(i)
            
                print(f"h = {ref_value} [м]" if env_PID.ctrl.use_ctrl else f"vartheta = {ref_value*180/pi} [град]")

                print(f'Расчет перехода с использованием ПИД-регулятора [{"h_func" if env_PID.ctrl.use_ctrl else "vartheta_func"}]')
                num_pid_interactions = int(env_PID.ctrl.tk/env_PID.ctrl.model.dt)
                self._test_env(ref_value, num_pid_interactions, env_PID, state0, manual=False, use_render=True, on_episode_end=callb)
                if storage is None:
                    storage = copy.deepcopy(tmp_storage)
                    if len(pid_coefs) > 1:
                        storage.set_suffix(name)
                else:
                    storage.merge(copy.deepcopy(tmp_storage), name)
                info_pid = dict(info)
                print(f"Характеристики ПП {name} | {coefs}:", info_pid)
                data = data.append({'Устройство': name, 'σ, [%]': info_pid['overshoot'],\
                    'tпп, [с]': info_pid['settling_time'], 'tв, [с]': info_pid['rise_time'],\
                        f"Δ, {'[м]' if env_PID.ctrl.use_ctrl else '[град]'}": info_pid['static_error'], 'Q, [-]': quality}, ignore_index=True)

            for model_name, env_init in env_inits.items():
                if no_neural:
                    break
                env = env_init() # создаем среду для тестирования
                tk = env.ctrl.tk
                num_interactions = int(tk/env.ctrl.sample_time)
                env.ctrl.use_storage = True
                self._init_names(model_name)

                print(f'Расчет перехода с использованием нейросетевого регулятора [{"h_func" if env.ctrl.use_ctrl else "vartheta_func"}]')
                self._test_env(ref_value, num_interactions, env, state0, manual=True, use_render=True, on_episode_end=callb)
                storage_neural = copy.deepcopy(tmp_storage)
                info_neural = dict(info)
                print(f"Характеристики ПП {model_name}:", info_neural)
                data = data.append({'Устройство': get_model_name_desc(model_name), 'σ, [%]': info_neural['overshoot'],\
                'tпп, [с]': info_neural['settling_time'], 'tв, [с]': info_neural['rise_time'],\
                    f"Δ, {'[м]' if env_PID.ctrl.use_ctrl else '[град]'}": info_neural['static_error'], 'Q, [-]': quality}, ignore_index=True)

                storage.merge(storage_neural, model_name)

            if plot:
                def get_pid_labels(label:str) -> List[str]:
                    if len(pid_coefs) > 1:
                        return [f"{label}__{pid_name_index(i)}" for i in range(len(pid_coefs))]
                    else:
                        return [label]
                storage.plot(["vartheta_ref", *get_pid_labels("vartheta"), *[f"vartheta__{model_name}" for model_name in model_names]], "t", 't, [с]', 'ϑ, [град]')
                if env_PID.ctrl.use_ctrl:
                    storage.plot(['hzh', *get_pid_labels("y"), *[f"h__{model_name}" for model_name in model_names]], "t", 't, [с]', 'h, [м]')
                storage.plot([*get_pid_labels("U_com"), *[f"{label}__{model_name}" for model_name in model_names for label in ["U_com", "U_PID"]]], 't', 't, [с]', 'Uком, [град]')
                storage.plot([*get_pid_labels("deltaz"), *[f"deltaz__{model_name}" for model_name in model_names]], "t", 't, [с]', 'δ, [град]')
                storage.plot([*get_pid_labels("rew"), *[f"rew__{model_name}" for model_name in model_names]], "t", 't, [с]', 'reward, [-]')

            if output_dir:
                model_dir = output_dir if (collect or len(env_inits) > 1) else os.path.join(output_dir, self.model_name)
                filepath = os.path.join(model_dir, f"data_{f'h_{ref_value}' if env_PID.ctrl.use_ctrl else f'vartheta_{ref_value*180/pi}'}.xlsx")
                os.makedirs(model_dir, exist_ok=True)
                data.set_index('Устройство', inplace=True)
                datas.append(data)
                data.to_excel(os.path.join(model_dir, f"data_{f'h_{ref_value}' if env_PID.ctrl.use_ctrl else f'vartheta_{ref_value*180/pi}'}_info.xlsx"),\
                    index=True, header=True)
                storages[filepath] = storage

        for i in range(len(datas)):
            datas[i]['σ, [%]'] = datas[i]['σ, [%]'].abs()
        data_mean = pd.concat(datas)
        data_mean = data_mean.groupby(data_mean.index).mean(False)
        data_mean.to_excel(os.path.join(model_dir, f"data_{'h' if env_PID.ctrl.use_ctrl else 'vartheta'}_info_mean.xlsx"),\
                    index=True, header=True)
        for filepath, storage in storages.items():
            storage.save(filepath, base="t")
        self._init_names(model_name_backup)


    def show(self) -> None:
        '''Показать структуру модели.'''
        if self.model is not None:
            attrs = ['gamma', 'max_grad_norm', 'gae_lambda', 'n_steps', 'learning_rate', 'ent_coef', 'vf_coef']
            print(self.model.policy.optimizer_class)
            for k in attrs:
                print(k, ':', self.model.__dict__[k])
            print('='*20)
            print(self.model.policy)
        else:
            print('Невозможно отобразить структуру модели: модель отсутствует.')