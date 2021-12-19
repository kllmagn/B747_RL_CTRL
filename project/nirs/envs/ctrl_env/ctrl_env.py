import numpy as np
import gym
from math import exp, pi, log
from gym import spaces

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
		if self.ctrl.manual_control: # ручное управление (с поддержкой ПИД-регулятора СУ)
			if self.n_actions:
				self.action_space = spaces.MultiDiscrete([self.n_actions]) 
			else:
				self.action_space = spaces.Box(low=-self.ctrl.delta_max, high=self.ctrl.delta_max, shape=(1,)) 
		else: # управление с поддержкой ПИД-регуляторов (наличие возможности коррекции коэффициентов)
			if self.n_actions:
				self.action_space = spaces.MultiDiscrete([self.n_actions]*8)
			else:
				self.action_space = spaces.Box(low=0.9, high=1.1, shape=(8,)) 
		self.observation_space = spaces.Box(low=-pi, high=pi, shape=(4,))
		self.state_box = np.zeros(self.observation_space.shape)
		self.reward_config = reward_config

	def _get_obs(self):
		# self.ctrl.model.state,\
		new_state = np.concatenate((np.array([self.ctrl.model.state_dict[k] for k in ['alpha', 'vartheta', 'wz']]),\
				np.array([self.ctrl.model.vartheta_ref]))) #, self.ctrl.vth_err.output()]))) #,\
				#self.ctrl.model.CXa, self.ctrl.model.CYa, self.ctrl.model.mz, self.ctrl.model.Kalpha, self.ctrl.model.dCm_ddeltaz, \
				#self.ctrl.err_vartheta, self.ctrl.calc_CS_err(), self.ctrl.model.hzh])))
		if len(self.observation_space.shape) > 1:
			self.state_box = np.vstack([self.state_box, new_state])
			self.state_box = np.delete(self.state_box, 0, 0)
		else:
			self.state_box = new_state
		return self.state_box

	def is_done(self):
		return self.ctrl.is_done #or (not self.is_testing and self.ctrl.is_limit_err)

	def get_reward(self, action):
		baseline = 1e-4
		Ay = self.reward_config.get('Ay', 1)
		Ae = 1-Ay
		max_y, max_e = 1*pi/80, 1*pi/180
		ky0 = self.reward_config.get('ky', log(baseline)/max_y)
		ke0 = self.reward_config.get('ke', log(baseline)/max_e)
		use_limit_punisher = False
		tp = 3
		ky = lambda t: ky0 #if t >= tp else ky0*t/tp
		ke = lambda t: ke0 #if t >= tp else ke0*t/tp
		if self.ctrl.manual_control:
			ry = Ay*exp(ky(self.ctrl.model.time)*(self.ctrl.model.vartheta_ref-self.ctrl.model.state_dict['vartheta'])**2) #self.ctrl.calc_SS_err())) #1/(1+ky*self.ctrl.calc_SS_err()**2)*exp(-self.ctrl.model.time/self.ctrl.tk)
			re = Ae*exp(ke(self.ctrl.model.time)*self.ctrl.model.state_dict['wz']**2) #-(self.ctrl.model.state_dict['wz']**2) #-self.ctrl.calc_SS_err()**2 #
			rl = -1 if (use_limit_punisher and self.ctrl.is_limit_err) else 0
			return ry+re+rl
		else:
			return 0.5/(self.ctrl.calc_CS_err()+1) + 0.5/(self.ctrl.calc_SS_err()+1)

	def step(self, action):
		if self.n_actions:
			if self.ctrl.manual_control:
				action = 2*action / (self.n_actions-1) - 1
				action *= self.ctrl.delta_max
			else:
				action = action / (self.n_actions-1) + 0.5
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
		#if self.ctrl.tk-self.ctrl.model.dt <= self.ctrl.model.time:
		#	print(self.ctrl.model.state_dict['vartheta']*180/pi, self.ctrl.model.vartheta_ref*180/pi)


def main():
	pass

if __name__ == '__main__':
	main()