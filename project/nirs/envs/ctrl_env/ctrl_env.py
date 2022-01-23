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
		is_testing:bool=False,
		reward_config:dict={},
		**ctrl_kwargs
		):
		super(ControllerEnv, self).__init__()
		self.ctrl = Controller(*ctrl_args, **ctrl_kwargs)
		self.is_testing = is_testing
		acts_low, acts_high = self._get_action_def()
		self.action_space = spaces.Box(low=np.array(acts_low), high=np.array(acts_high), shape=(len(acts_low),)) 
		_, _, norms, n = self._get_obs_def()
		self.observation_space = spaces.Box(
			low=np.array([norm[0] for norm in norms]),
			high=np.array([norm[1] for norm in norms]),
			shape=(n,)
		)
		self.state_box = np.zeros(self.observation_space.shape)
		self.reward_config = reward_config

	def _get_action_def(self):
		acts_low, acts_high = [], []
		if self.ctrl.use_ctrl:
			if self.ctrl.model.use_PID_CS:
				if not self.ctrl.full_auto:
					acts_low.extend([0.9]*4)
					acts_high.extend([1.1]*4)
			else:
				acts_low.extend([-self.ctrl.vartheta_max])
				acts_high.extend([self.ctrl.vartheta_max])
		if self.ctrl.model.use_PID_SS:
			if not self.ctrl.full_auto:
				acts_low.extend([0.9]*4)
				acts_high.extend([1.1]*4)
		else:
			acts_low.extend([-self.ctrl.delta_max])
			acts_high.extend([self.ctrl.delta_max])
		return acts_low, acts_high

	def _get_obs_def(self):
		ks = []
		add = []
		norms = []

		ks.extend(['alpha', 'vartheta', 'wz'])
		norms.extend([(-pi, pi), (-pi, pi), (-pi, pi)])
		add.extend([self.ctrl.vartheta_ref, self.ctrl.err_vartheta])
		norms.extend([(-pi, pi), (-pi, pi)])

		'''
		if self.ctrl.use_ctrl:
			ks.extend(['y'])
			norms.extend([(0, 80000)])
			add.extend([self.ctrl.model.hzh, self.ctrl.err_h])
			norms.extend([(0, 80000), (-80000, 80000)])
		'''

		return ks, add, norms, len(ks)+len(add)

	def _get_obs(self):
		# self.ctrl.model.state,\
		ks, add, _, _ = self._get_obs_def()
		new_state = np.concatenate((np.array([self.ctrl.model.state_dict[k] for k in ks]), np.array(add)))
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
			Av = self.reward_config.get('Av', 0.55)
			Adeltaz = self.reward_config.get('Adeltaz', 0.4)
			kv = self.reward_config.get('kv', 180/pi)
			kw = self.reward_config.get('kw', 1)
			k1 = self.reward_config.get('k1', 100)
			k2 = self.reward_config.get('k2', 1)
			dvartheta = self.reward_config.get('dvartheta', 20*pi/180)
			Aw = 1 - Av - Adeltaz
			use_limit_punisher = True
			rv = Av/(1+kv*abs(self.ctrl.err_vartheta))
			rdeltaz = Adeltaz/(1+k2*(abs(self.ctrl.err_vartheta)/dvartheta)*abs(self.ctrl.deltaz-self.ctrl.model.deltaz_ref))
			rw = Aw/(1+kw*abs(self.ctrl.model.state_dict['wz'])/(1+k1*abs(self.ctrl.err_vartheta)/(20*pi/180)))
			rl = -2 if (use_limit_punisher and self.ctrl.is_limit_err) else 0
			#print('rv:', rv, 'rdeltaz:', rdeltaz, 'rw:', rw)
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