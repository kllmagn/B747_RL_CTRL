import numpy as np
import gym
from math import exp, pi, log
from gym import spaces

from .tools import calc_err
from .ctrl import Controller

class ControllerEnv(gym.Env):
	"""Среда взаимодействия между контроллером и нейросетевой моделью."""
	metadata = {'render.modes': ['human']}
	def __init__(
		self,
		*ctrl_args,
		n_actions:int=None,
		is_testing:bool=False,
		reward_config:dict={},
		**ctrl_kwargs
		):
		super(ControllerEnv, self).__init__()
		self.ctrl = Controller(*ctrl_args, **ctrl_kwargs)
		self.n_actions = n_actions
		self.is_testing = is_testing
		if self.ctrl.full_auto or self.ctrl.manual_stab: # ручное управление (с поддержкой ПИД-регулятора СУ)
			if self.n_actions:
				self.action_space = spaces.MultiDiscrete([self.n_actions]) 
			else:
				self.action_space = spaces.Box(low=-self.ctrl.delta_max, high=self.ctrl.delta_max, shape=(1,)) 
		else: # управление с поддержкой ПИД-регуляторов (наличие возможности коррекции коэффициентов)
			if self.n_actions:
				self.action_space = spaces.MultiDiscrete([self.n_actions]*8)
			else:
				self.action_space = spaces.Box(low=0.9, high=1.1, shape=(8,)) 
		self.observation_space = spaces.Box(low=-pi, high=pi, shape=(5,))
		self.state_box = np.zeros(self.observation_space.shape)
		self.reward_config = reward_config

	def _get_obs(self):
		# self.ctrl.model.state,\
		new_state = np.concatenate((np.array([self.ctrl.model.state_dict[k] for k in ['alpha', 'vartheta', 'wz']]),\
				np.array([self.ctrl.vartheta_ref, self.ctrl.err_vartheta]))) #, (self.ctrl.model.deltaz if self.ctrl.manual_stab else self.ctrl.model.deltaz_ref)]))) #, self.ctrl.vartheta_ref])))
				#self.ctrl.model.CXa, self.ctrl.model.CYa, self.ctrl.model.mz, self.ctrl.model.Kalpha, self.ctrl.model.dCm_ddeltaz, \
				#self.ctrl.err_vartheta, self.ctrl.calc_CS_err(), self.ctrl.model.hzh])))
		if len(self.observation_space.shape) > 1:
			self.state_box = np.vstack([self.state_box, new_state])
			self.state_box = np.delete(self.state_box, 0, 0)
		else:
			self.state_box = new_state
		return self.state_box

	def is_done(self):
		return self.ctrl.is_done or (not self.is_testing and self.ctrl.is_limit_err)

	def get_reward(self, action):
		baseline = 1e-4
		mode = self.reward_config.get('mode', 'standard')
		if mode == 'standard':
			Av = self.reward_config.get('Av', 0.2)
			Aw = 1-Av
			max_ve, max_we = (5*pi/180)**2, (0.001*pi/180/self.ctrl.model.dt)**2
			kv0 = self.reward_config.get('kv', log(baseline)/max_ve)
			kw0 = self.reward_config.get('kw', log(baseline)/max_we)
			use_limit_punisher = True
			tp = 0
			kv = lambda t: kv0 if t >= tp else kv0*t/tp
			kw = lambda t: kw0 if t >= tp else kw0*t/tp
			rv = Av/(1+180/pi*abs(self.ctrl.err_vartheta)) #Av*exp(kv(self.ctrl.model.time)*(self.ctrl.model.state_dict['vartheta']-self.ctrl.vartheta_ref)**2) #self.ctrl.calc_SS_err())) #1/(1+ky*self.ctrl.calc_SS_err()**2)*exp(-self.ctrl.model.time/self.ctrl.tk)
			rdeltaz = 0.8/(1+abs(self.ctrl.deltaz-self.ctrl.model.deltaz_ref))
			rw = Aw*exp(kw(self.ctrl.model.time)*self.ctrl.model.state_dict['wz']**2) #-(self.ctrl.model.state_dict['wz']**2) #-self.ctrl.calc_SS_err()**2 #
			rl = -2 if (use_limit_punisher and self.ctrl.is_limit_err) else 0
			#print('rv:', rv, 'rdeltaz:', rdeltaz)
			return rv+rdeltaz+rw+rl
		elif mode == 'VRS': # velocity reward strategy
			e = self.ctrl.err_vartheta
			dedt = self.ctrl.deriv_dict.output('dvedt')
			dedt_prev = self.ctrl.memory_dict.output('dvedt')
			rv = -dedt if e > 0 else dedt
			rt = -rv if (np.sign(dedt) != np.sign(dedt_prev)) and abs(dedt) < abs(dedt_prev) else rv
			#print(self.ctrl.vartheta_ref*180/pi, self.ctrl.model.state_dict['vartheta']*180/pi, self.ctrl.err_vartheta*180/pi)
			return rt
		else:
			#return 1/(1+180/pi*(self.ctrl.model.state_dict['vartheta']-self.ctrl.vartheta_ref)**2) + (-2 if (use_limit_punisher and self.ctrl.is_limit_err) else 0)
			return 0.5/(self.ctrl.calc_CS_err()+1) + 0.5/(self.ctrl.calc_SS_err()+1)

	def step(self, action):
		if self.n_actions:
			if self.ctrl.manual_stab:
				action = 2*action / (self.n_actions-1) - 1
				action *= self.ctrl.delta_max
			else:
				action = action / (self.n_actions-1) + 0.5
		self.ctrl.memory_dict.input('dvedt', self.ctrl.deriv_dict.output('dvedt'))
		self.ctrl.step(action)
		observation = self._get_obs()
		reward = self.get_reward(action)
		done = self.is_done()
		info = {} #'storage': self.ctrl.storage}
		return observation, reward, done, info

	def reset(self):
		self.ctrl.reset()
		self.state_box = np.zeros(self.observation_space.shape)
		observation = self._get_obs()
		return observation

	def render(self, mode='human'):
		pass


def main():
	pass

if __name__ == '__main__':
	main()