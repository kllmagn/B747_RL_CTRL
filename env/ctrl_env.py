import random
import optuna
from typing import Callable, Tuple
from core.controller import Controller, CtrlMode

import numpy as np
import gym
import torch as th
from math import exp, pi
from gym import spaces
from enum import Enum

from tools.general import calc_exp_k, calc_stepinfo


class ObservationType(Enum):
	PID_LIKE = 0 # метод подобия
	SPEED_MODE = 1 # учет скоростного режима
	PID_AERO = 2
	PID_SPEED_AERO = 3
	MODEL_STATE = 4

class RewardType(Enum):
	CLASSIC = 0
	PID_LIKE = 1
	QUALITY = 2
	MINIMAL = 3
	TF_REFERENCE = 4

def get_rew_config(rew_type:RewardType, trial:optuna.Trial):
	if rew_type == RewardType.CLASSIC:
		return {
			'k1': trial.suggest_uniform('k1', 0.1, 1),
			'k2': trial.suggest_uniform('k2', 0.1, 1),
			'k3': trial.suggest_uniform('k3', 0.1, 1),
			'k0': trial.suggest_uniform('k0', 1.0, 10.0),
			'kITSE': trial.suggest_uniform('kITSE', 0.01, 10.0),
			'kf': trial.suggest_uniform('kf', 0.05, 10.0),
		}
	elif rew_type == RewardType.PID_LIKE:
		return {
			'k': trial.suggest_uniform('k', 1, 20)
		}
	elif rew_type == RewardType.MINIMAL:
		return {
			'rmax': trial.suggest_uniform('rmax', 0, 1.0),
			'k1': trial.suggest_uniform('k1', 0.1, 5.0),
			'k2': trial.suggest_uniform('k2', 0.1, 5.0),
		}
	elif rew_type == RewardType.TF_REFERENCE:
		return {
			'k1': trial.suggest_uniform('k1', 0.1, 5.0),
			'k2': trial.suggest_uniform('k1', 0.1, 5.0),
		}
	else:
		raise ValueError(f"Неподдерживаемый тип награды для оптимизации конфигурации: {rew_type}.")


class ControllerEnv(gym.Env):
	"""Среда взаимодействия между контроллером и нейросетевой моделью."""
	metadata = {'render.modes': ['human']}

	def __init__(
		self,
		observation_type:ObservationType,
		reward_type:RewardType,
		norm_obs:bool,
		norm_act:bool,
		*ctrl_args, # аргументы контроллера
		**ctrl_kwargs
		):
		super(ControllerEnv, self).__init__()

		random.seed(1) # выставляем seed для детерминированного поведения
		np.random.seed(0) # выставляем seed для детерминированного поведения
		th.manual_seed(1) # выставляем seed для детерминированного поведения

		self.observation_type = observation_type # тип используемого вектора состояния
		self.reward_type = reward_type # тип используемой функции подкрепления
		self.norm_obs = norm_obs
		self.norm_act = norm_act

		ControllerEnv.get_reward = self._get_reward_def(self.reward_type)
		ControllerEnv._get_obs_raw = self._create_obs()
		ControllerEnv._get_obs_def = self._create_obs_def()

		self.ctrl = Controller(*ctrl_args, **ctrl_kwargs) # объект контроллера

		acts_low, acts_high = self._get_action_def() # получить определение действия
		assert acts_low.shape == acts_high.shape, 'Размерности граничных определений действия не совпадают.'
		obs_low, obs_high = self._get_obs_def() # получить определение вектора состояния
		assert obs_low.shape == obs_high.shape, 'Размерности граничных определений вектора состояния не совпадают.'

		if self.norm_act:
			self.action_space = spaces.Box(low=-1, high=1, shape=acts_low.shape)
		else:
			self.action_space = spaces.Box(low=acts_low, high=acts_high, shape=acts_low.shape)
		if self.norm_obs:
			self.observation_space = spaces.Box(low=-1, high=1, shape=obs_low.shape)
		else:
			self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_low.shape)
		self.state_box = np.zeros(self.observation_space.shape)


	def _get_reward_def(self, reward_type:RewardType, reward_config:dict={}) -> Callable[['ControllerEnv', np.ndarray], float]:
		'''Получить определение функции награды среды.'''
		if reward_type == RewardType.CLASSIC:
			k1, k2, k3 = reward_config.get('k1', 2), reward_config.get('k2', 2), reward_config.get('k3', 1)
			k0 = calc_exp_k(0.8, 0.3) #reward_config.get('k0', 2) #2+self.ctrl.model.time # раньше был 1
			kf = reward_config.get('kf', 0.1)
			kITSE = reward_config.get('kITSE', 6)
			kt = calc_exp_k(0.8, 10)
			ko = calc_exp_k(0.75, 0.15)
			s = k1 + k2 + k3
			k1 /= s
			k2 /= s
			k3 /= s
			k2_0 = 1 #1/(1+0.5*abs(self.ctrl.model.dvartheta)/abs(2*vf))
			def rew(self:ControllerEnv, action:np.ndarray) -> float:
				vf = self.ctrl.vartheta_ref if self.ctrl.vartheta_ref else self.ctrl.vartheta_max # требуемое значение угла тангажа
				# компонент ошибки стабилизации
				r1 = 0.1*exp(-k0*(k1*abs(self.ctrl.model.dvartheta)+k2*k2_0*abs(self.ctrl.model.dvartheta_dt)+k3*abs(self.ctrl.model.dvartheta_dt_dt))/abs(vf)) 
				# компонент перерегулирования
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0: # если перерегулирование сверху
					r2 = 0.1*exp(-ko*abs(self.ctrl.model.dvartheta/vf)) # чем больше перерегулирование сверху, тем меньше награда
				else:
					r2 = 0.1
				# компонент времени ПП
				if abs(self.ctrl.model.dvartheta/vf) > 0.05: # если процесс вышел за допустимые пределы по времени ПП
					r3 = 0.1*exp(-kt*self.ctrl.model.time) # чем дольше идет процесс, тем меньше награда
				else:
					r3 = 0.1
				# компонент интегральной временной квадратичной ошибки ПП
				r4 = 0.7*exp(-kITSE*self.ctrl.model.ITSE/(20*vf**2))
				# формирующий компонент
				rf = -kf*abs(self.ctrl.model.dvartheta/(2*vf))*(abs(action[0]-self.ctrl.model.deltaz_ref))/(34*pi/180) if self.ctrl.ctrl_mode == CtrlMode.DIRECT_CONTROL else 0
				r = r1 + r2 + r3 + r4 + rf # полное значение функции подкрепления
				return r
		elif reward_type == RewardType.PID_LIKE:
			k = reward_config.get('k', 10)
			def rew(self:ControllerEnv, _:np.ndarray):
				r = exp(-k*abs(self.ctrl.model.deltaz_com-self.ctrl.model.deltaz_ref)/(34*pi/180))
				return r
		elif reward_type == RewardType.QUALITY:
			def rew(self:ControllerEnv, _:np.ndarray):
				r = self.ctrl.quality()
				return r
		elif reward_type == RewardType.MINIMAL:
			rmax = reward_config.get('rmax', 0.2)
			Qmax = 1 #-rmax
			k1 = reward_config.get('k1', 2)
			k2 = reward_config.get('k2', 0.5)
			def rew(self:ControllerEnv, _:np.ndarray) -> float:
				vf = self.ctrl.vartheta_ref if self.ctrl.vartheta_ref else self.ctrl.vartheta_max # требуемое значение угла тангажа
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0: # если перерегулирование сверху
					kovershoot = exp(-k1*abs(self.ctrl.model.dvartheta/vf)) # чем больше перерегулирование сверху, тем меньше награда
				else:
					kovershoot = 1.
				if abs(self.ctrl.model.dvartheta/vf) > 0.05: # если процесс вышел за допустимые пределы по времени ПП
					ktp = exp(-k2*self.ctrl.model.time) # чем дольше идет процесс, тем меньше награда
				else:
					ktp = 1.
				Q = Qmax*self.ctrl.quality() #exp(-2*self.ctrl.model.TSE/(self.ctrl.vartheta_ref**2*self.ctrl.tk))
				r = rmax*kovershoot*ktp
				R = Q # + r
				return R
		elif reward_type == RewardType.TF_REFERENCE:
			overshoot_ref = reward_config.get('overshoot_ref', 2)
			tp_ref = reward_config.get('tp_ref', 5)
			k = reward_config.get('k', 0.1)
			tp = 0
			def rew(self:ControllerEnv, _:np.ndarray):
				nonlocal tp
				vf = self.ctrl.vartheta_ref if self.ctrl.vartheta_ref else self.ctrl.vartheta_max # требуемое значение угла тангажа
				overshoot = abs(self.ctrl.model.dvartheta/vf)*100
				if overshoot > 5:
					tp = self.ctrl.model.time
				r = exp(-k*abs(overshoot-overshoot_ref)*abs(tp_ref-tp))
				return r
		else:
			raise ValueError("Неподдерживаемый тип функции подкрепления: ", reward_type)
		return rew


	def _get_action_def(self) -> Tuple[np.ndarray, np.ndarray]:
		'''Получить определение управления (предельные величины).'''
		return np.array([-self.ctrl.action_max]), np.array([self.ctrl.action_max])


	def _create_obs_def(self) -> Callable[['ControllerEnv'], Tuple[np.ndarray, np.ndarray]]:
		'''Получить определение вектора состояния среды (предельные величины вектора).'''
		if self.observation_type == ObservationType.PID_LIKE:
			obs_max = np.array([60*pi, pi, pi])
		elif self.observation_type == ObservationType.SPEED_MODE:
			obs_max = np.array([60*pi, pi, pi, 500, 100])
		elif self.observation_type == ObservationType.PID_SPEED_AERO:
			obs_max = np.array([60*pi, pi, pi, 500, 100, 0.5, 2, 0.6, 0.05, 1.])
		elif self.observation_type == ObservationType.PID_AERO:
			obs_max = np.array([60*pi, pi, pi, 0.5, 2, 0.6, 0.05, 1.])
		elif self.observation_type == ObservationType.MODEL_STATE:
			obs_max = np.array([10*pi/180, 12000, 15000, 500, 100, pi, pi])
		else:
			raise ValueError("Неподдерживаемый тип вектора состояния среды: ", self.observation_type)
		return lambda _: (-obs_max, obs_max)


	def _create_obs(self) -> Callable[['ControllerEnv'], np.ndarray]:
		'''Сформировать функцию получения вектора состояния среды (в зависимости от режима).'''
		if self.observation_type == ObservationType.PID_LIKE:
			obs = lambda self: np.array([self.ctrl.model.dvartheta_int, self.ctrl.model.dvartheta, self.ctrl.model.dvartheta_dt])
		elif self.observation_type == ObservationType.SPEED_MODE:
			obs = lambda self: np.array([self.ctrl.model.dvartheta_int, self.ctrl.model.dvartheta, self.ctrl.model.dvartheta_dt, self.ctrl.model.state_dict['Vx'], self.ctrl.model.state_dict['Vy']])
		elif self.observation_type == ObservationType.PID_SPEED_AERO:
			obs = lambda self: np.array([self.ctrl.model.dvartheta_int, self.ctrl.model.dvartheta, self.ctrl.model.dvartheta_dt,\
				self.ctrl.model.state_dict['Vx'], self.ctrl.model.state_dict['Vy'],\
					self.ctrl.model.CXa, self.ctrl.model.CYa, self.ctrl.model.mz, self.ctrl.model.dCm_ddeltaz, self.ctrl.model.Kalpha])
		elif self.observation_type == ObservationType.PID_AERO:
			obs = lambda self: np.array([self.ctrl.model.dvartheta_int, self.ctrl.model.dvartheta, self.ctrl.model.dvartheta_dt,\
				self.ctrl.model.CXa, self.ctrl.model.CYa, self.ctrl.model.mz, self.ctrl.model.dCm_ddeltaz, self.ctrl.model.Kalpha])
		elif self.observation_type == ObservationType.MODEL_STATE:
			obs = lambda self: np.array([self.ctrl.vartheta_ref, *self.ctrl.model.state])
		else:
			raise ValueError(f"Неподдерживаемый режим представления вектора состояния среды: {self.observation_type}")
		return obs


	def _get_obs(self):
		obs = self._get_obs_raw()
		if self.norm_obs:
			_, obs_max = self._get_obs_def()
			obs /= obs_max
		if len(self.observation_space.shape) > 1:
			self.state_box = np.vstack([self.state_box, obs])
			self.state_box = np.delete(self.state_box, 0, 0)
		else:
			self.state_box = obs
		return self.state_box


	def set_rew_config(self, rew_config:dict):
		'''Установить конфигурацию функции подкрепления среды.'''
		ControllerEnv.get_reward = self._get_reward_def(self.reward_type, rew_config)


	def is_done(self):
		'''Является ли процесс моделирования оконченым.'''
		return self.ctrl.is_done or self.ctrl.is_nan_err or self.ctrl.is_limit_err


	def step(self, action):
		'''Произвести один шаг симуляции среды.'''
		if self.norm_act and action is not None:
			_, action_max = self._get_action_def()
			action *= action_max
		self.ctrl.step(action)
		observation = self._get_obs()
		reward = self.get_reward(action)
		done = self.is_done()
		info = {} #'storage': self.ctrl.storage}
		return observation, reward, done, info


	def reset(self, state0:np.ndarray=None):
		'''Выполнить сброс среды.'''
		self.ctrl.reset(state0=state0)
		self.state_box = np.zeros(self.observation_space.shape)
		observation = self._get_obs()
		return observation


	def render(self, mode='human'):
		pass
