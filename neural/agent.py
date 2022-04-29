import os
import copy
from math import pi
import random
from threading import Thread
from matplotlib.pyplot import plot

import numpy as np

from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from .callbacks import *
from tensorboard import program

from tqdm import tqdm

import torch as th

import optuna

import onnx
import onnxruntime as ort

from env.ctrl_env import *
from .pretrain import pretrain_agent_imit
from .setups import hyperparams, trial_hyperparams, TrainPlotter

class ControllerAgent:

    def __init__(self, net_class=A2C, use_tb=False, log_dir='./.logs', model_name='best_model'):
        random.seed(1)
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
        self.model_name = model_name
        self.bm_name = f'{model_name}.zip' # выставляем имя файла модели

        self.callbacks = [] # обязательные Callback функции

    
    def _init_callbacks(self, verbose:int=0):
        '''
        Инициализировать обязательные Callback функции.
        '''
        self.callbacks = []


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
                load_path = os.path.join(self.log_dir, self.bm_name)
                save_path = os.path.join(self.log_dir, 'optimization', self.bm_name)
                os.replace(save_path, load_path)
        def objective(trial:optuna.Trial):
            if opt_hp:
                hp = trial_hyperparams(self.net_class, trial, self.hp)
            else:
                hp = self.hp
            reward_config = {
                'k1': trial.suggest_categorical('k1', [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0]),
                'k2': trial.suggest_categorical('k2', [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0]),
                'k3': trial.suggest_categorical('k3', [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0]),
            }
            env = ControllerEnv(*ctrl_env_args, reward_config=reward_config, **ctrl_env_kwargs)
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
            #sobtr = SaveOnBestTrainingRewardCallback(self.model_name, 10000, savebest_dir, 1)
            #cb2 = SaveOnBestQualityMetricCallback(lambda env: env.get_attr('ctrl')[0].model.TAE, 'TAE', 10000, log_dir=savebest_dir, maximize=opt_max) #vth_err_abs.output()
            tf_custom_recorder = CustomTransferProcessRecorder(
                env_gen=lambda:self._wrap_env(ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs), use_monitor=False),
                vartheta_ref=5*pi/180,
                state0=[0, 11000, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                log_interval=5000,
                filename=self.bm_name,
                log_dir=savebest_dir,
                window_length=5,
                verbose=0)
            es = EarlyStopping(lambda: tf_custom_recorder.mean_quality, 'mean_quality', 10000, 4, verbose=1, startup_step=es_startup, maximize=opt_max)
            with ProgressBarManager(total_timesteps) as callback:
                cb_list = CallbackList([callback, tf_custom_recorder])
                self.model.learn(total_timesteps=total_timesteps, callback=cb_list)
            self.model = self.net_class.load(os.path.join(savebest_dir, self.bm_name))
            return tf_custom_recorder.best_mean_quality

        study = optuna.create_study(direction=("maximize" if opt_max else "minimize"))
        study.optimize(objective, n_trials=500, callbacks=[save_model_callback], catch=(ValueError,))
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


    def train(
            self,
            *ctrl_env_args,
            timesteps=50000,
            preload=False,
            use_es=True,
            optimize=False,
            opt_max=True,
            opt_hp=True,
            verbose:int=1,
            log_interval:int=1000,
            show_plotter:bool=False,
            **ctrl_env_kwargs
        ):
        '''
        Произвести обучение нейросетевой модели.
        '''
        self.env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(self.env)
        if optimize:
            print('Оптимизирую с помощью Optuna')
            self.optimize(timesteps, opt_max=opt_max, opt_hp=opt_hp, *ctrl_env_args, **ctrl_env_kwargs)
            self.model = self.net_class.load(os.path.join(self.log_dir, self.bm_name), tensorboard_log=self.tb_log, verbose=verbose)
            self.model.set_env(env)
        else:
            if preload:
                preload_path = os.path.join(self.log_dir, self.bm_name)
                print('Предзагружаю модель из', preload_path)
                self.model = self.net_class.load(preload_path, tensorboard_log=self.tb_log, verbose=verbose)
                self.model.set_env(env)
            else:
                print('Создаю новую модель:', str(self.net_class))
                self.model = self.net_class('MlpPolicy', env, tensorboard_log=self.tb_log, verbose=verbose, **self.hp)
        cb1 = SaveOnBestTrainingRewardCallback(self.bm_name, 5000, self.log_dir, 1)
        cb_metric = SaveOnBestQualityMetricCallback(self.bm_name, lambda env: env.get_attr('ctrl')[0].deltaz_diff_int.output(), 'deltaz_diff_int', 10000, log_dir=self.log_dir, maximize=False)
        cb2 = EarlyStopping(lambda: cb1.mean_reward, 'vth_err', 10000, 4, verbose=1, maximize=True)
        def transfer_quality(env):
            info = env.get_attr('ctrl')[0].stepinfo_SS(use_backup=True)
            time, overshoot = info['settling_time'], info['overshoot']
            print('Время ПП:', time, 'Перерегулирование:', overshoot)
            if time is None or overshoot is None:
                return np.inf
            else:
                return time*abs(overshoot)
        transfer_callback = SaveOnBestQualityMetricCallback(
            self.bm_name,
            transfer_quality,
            'tf_quality',
            -1,
            self.log_dir,
            verbose=1,
            maximize=False,
            mean_num = 10
        )

        tf_custom_recorder = CustomTransferProcessRecorder(
            env_gen=lambda:self._wrap_env(ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs), use_monitor=False),
            vartheta_ref=5*pi/180,
            state0=[0, 11000, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            log_interval=5000,
            filename=self.bm_name,
            log_dir=self.log_dir,
            window_length=100,
            verbose=1)

        cbs = [tf_custom_recorder] #, cb1]
        if use_es:
            cbs.append(cb2)
        cb = CallbackList(cbs)

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


    def test(self, *ctrl_env_args, ht_func=None, varthetat_func=None, plot=False, **ctrl_env_kwargs):
        ctrl_env_kwargs['random_reset'] = False

        self.env = ControllerEnv(*ctrl_env_args, h_func=ht_func, vartheta_func=varthetat_func, use_storage=True,\
            is_testing=True, **ctrl_env_kwargs)

        tk = ctrl_env_kwargs['tk']
        print('Расчет перехода с использованием нейросетевого регулятора [func]')
        env = self._wrap_env(self.env, manual_reset=True)
        hf = vf = info = None
        def callb(env):
            nonlocal vf, hf, info
            vf = env.get_attr('ctrl')[0].vartheta_func
            hf = env.get_attr('ctrl')[0].h_func
            ctrl_obj = env.get_attr('ctrl')[0]
            if ctrl_obj.use_ctrl:
                info = ctrl_obj.stepinfo_CS()
            else:
                info = ctrl_obj.stepinfo_SS()
            print(info)
            print('Суммарная ошибка по углу:', ctrl_obj.vth_err.output())
        num_interactions = int(tk/self.env.ctrl.sample_time)
        mean_reward, std_reward, storage1 = self.test_env(num_interactions, env, use_render=True, on_episode_end=callb)
        info_neural = dict(info)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")

        print('Расчет перехода с использованием ПИД-регулятора [func]')
        ctrl_env_kwargs['h_func'] = hf
        ctrl_env_kwargs['vartheta_func'] = vf
        ctrl_env_kwargs['no_correct'] = ctrl_env_kwargs['use_storage'] = ctrl_env_kwargs['is_testing'] = True
        ctrl_env_kwargs['manual_ctrl'] = ctrl_env_kwargs['manual_stab'] = False
        ctrl_env_kwargs['tk'] = tk
        del ctrl_env_kwargs['sample_time']
        env = ControllerEnv(*ctrl_env_args, **ctrl_env_kwargs)
        env = self._wrap_env(env, manual_reset=True)
        mean_reward, std_reward, storage2 = self.test_env(\
            num_interactions*int(self.env.ctrl.sample_time/self.env.ctrl.model.dt),\
                env, no_action=True, use_render=True, on_episode_end=callb)
        info_pid = dict(info)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")

        storage2.merge(storage1, 'neural')
        
        if plot:
            storage2.plot(["vartheta_ref", "vartheta_ref_neural", "vartheta_neural", "vartheta"], "t", 't, [с]', 'ϑ, [град]')
            if env.envs[0].env.ctrl.use_ctrl:
                storage2.plot(['hzh', 'hzh_neural', 'y_neural', 'y'], "t", 't, [с]', 'h, [м]')
            storage2.plot(['deltaz_neural', 'deltaz', 'deltaz_ref_neural'], 't', 't, [с]', 'δ_ком, [град]')
            storage2.plot(["deltaz_real_neural", "deltaz_real"], "t", 't, [с]', 'δ, [град]')

        return storage2, info_neural, info_pid


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