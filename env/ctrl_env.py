import random
from typing import Callable, Tuple
from core.controller import Controller, CtrlMode

import numpy as np
import gym
import torch as th
from math import exp, pi
from gym import spaces
from enum import Enum


class ObservationType(Enum):
	PID_LIKE = 0 # метод подобия
	SPEED_MODE = 1 # учет скоростного режима
	PID_AERO = 2
	PID_SPEED_AERO = 3
	MODEL_STATE = 4

class RewardType(Enum):
	CLASSIC = 0
	TAE = 1
	TSE = 2
	PID_LIKE = 3

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


	def _get_reward_def(self, reward_type:RewardType) -> Callable[['ControllerEnv', np.ndarray], float]:
		if reward_type == RewardType.CLASSIC:
			def rew(self:ControllerEnv, action:np.ndarray) -> float:
				vf = self.ctrl.vartheta_ref if self.ctrl.vartheta_ref else self.ctrl.vartheta_max
				k1, k2, k3 = 2, 2, 1
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0:
					# перерегулирование с противоположной стороны
					self.dv_max = max(self.dv_max, self.ctrl.model.dvartheta) if self.ctrl.model.dvartheta > 0 else min(self.dv_max, self.ctrl.model.dvartheta)
				s = k1 + k2 + k3
				k1 /= s
				k2 /= s
				k3 /= s
				k2_0 = 1/(1+0.5*abs(self.ctrl.model.dvartheta)/abs(2*vf))
				r = exp(-(k1*abs(self.ctrl.model.dvartheta)+k2*k2_0*abs(self.ctrl.model.dvartheta_dt)+k3*abs(self.ctrl.model.dvartheta_dt_dt))/abs(2*vf)) #-abs(self.dv_max)
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0:
					r *= exp(-0.5*abs(self.ctrl.model.dvartheta/(2*self.ctrl.vartheta_ref)))
				k = abs(self.ctrl.model.dvartheta/(2*self.ctrl.vartheta_max))
				rf = -k*(abs(action[0]-self.ctrl.model.deltaz_ref))/(34*pi/180) if self.ctrl.ctrl_mode == CtrlMode.DIRECT_CONTROL else 0
				r = r + rf
				return r
		elif reward_type == RewardType.TAE:
			def rew(self:ControllerEnv, action:np.ndarray):
				r = -self.ctrl.model.TAE
				return r
		elif reward_type == RewardType.TSE:
			def rew(self:ControllerEnv, action:np.ndarray):
				r = -self.ctrl.model.TSE
				return r
		elif reward_type == RewardType.PID_LIKE:
			def rew(self:ControllerEnv, action:np.ndarray):
				r = exp(-10*abs(self.ctrl.model.deltaz_com-self.ctrl.model.deltaz_ref)/(34*pi/180))
				return r
		else:
			raise ValueError("Неподдерживаемый тип функции подкрепления: ", reward_type)
		return rew


	def _get_action_def(self) -> Tuple[np.ndarray, np.ndarray]:
		return np.array([-self.ctrl.action_max]), np.array([self.ctrl.action_max])


	def _create_obs_def(self) -> Callable[['ControllerEnv'], Tuple[np.ndarray, np.ndarray]]:
		if self.observation_type == ObservationType.PID_LIKE:
			obs_max = np.array([6000, pi, pi])
		elif self.observation_type == ObservationType.SPEED_MODE:
			obs_max = np.array([6000, pi, pi, 500, 100])
		elif self.observation_type == ObservationType.PID_SPEED_AERO:
			obs_max = np.array([6000, pi, pi, 500, 100, 0.5, 2, 0.6, 0.05, 1.])
		elif self.observation_type == ObservationType.PID_AERO:
			obs_max = np.array([6000, pi, pi, 0.5, 2, 0.6, 0.05, 1.])
		elif self.observation_type == ObservationType.MODEL_STATE:
			obs_max = np.array([10*pi/180, 12000, 15000, 500, 100, pi, pi])
		else:
			raise ValueError("Неподдерживаемый тип вектора состояния среды: ", self.observation_type)
		return lambda _: (-obs_max, obs_max)


	def _create_obs(self) -> Callable[['ControllerEnv'], np.ndarray]:
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


	def is_done(self):
		return self.ctrl.is_done or self.ctrl.is_nan_err or self.ctrl.is_limit_err


	def step(self, action):
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
		self.dv_max = 0
		self.ctrl.reset(state0=state0)
		self.state_box = np.zeros(self.observation_space.shape)
		observation = self._get_obs()
		return observation


	def render(self, mode='human'):
		pass
