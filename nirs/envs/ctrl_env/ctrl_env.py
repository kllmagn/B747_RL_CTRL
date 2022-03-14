import numpy as np
import gym
from math import exp, pi, log
from gym import spaces
from pandas import array

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
		_, _, norms, shape = self._get_obs_def()
		self.observation_space = spaces.Box(
			low=np.array([norm[0] for norm in norms]),
			high=np.array([norm[1] for norm in norms]),
			shape=shape
		)
		self.state_box = np.zeros(self.observation_space.shape)
		self.reward_config = reward_config

		self.action_prev = None

	def _get_action_def(self):
		acts_low, acts_high = [], []
		if self.ctrl.use_ctrl:
			if self.ctrl.model.use_PID_CS:
				if not self.ctrl.no_correct:
					acts_low.extend([0.9]*4)
					acts_high.extend([1.1]*4)
			else:
				acts_low.extend([-self.ctrl.vartheta_max])
				acts_high.extend([self.ctrl.vartheta_max])
		if self.ctrl.model.use_PID_SS:
			if not self.ctrl.no_correct:
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

		#ks.extend(['alpha', 'vartheta', 'wz'])
		#norms.extend([(-pi, pi), (-pi, pi), (-pi, pi)])
		add.extend([self.ctrl.model.dvartheta_int, self.ctrl.model.dvartheta, self.ctrl.model.dvartheta_dt])
		norms.extend([(-np.inf, np.inf), (-pi, pi), (-pi, pi)])
		#add.extend([self.ctrl.err_vartheta, self.ctrl.vartheta_ref])
		#norms.extend([(-pi, pi), (-pi, pi)])
		#add.extend([self.ctrl.model.deltaz])
		#add.extend([self.ctrl.model.deltaz_real])
		#norms.extend([(-self.ctrl.delta_max, self.ctrl.delta_max)])
		#norms.extend([(-self.ctrl.delta_max, self.ctrl.delta_max)])

		#ks.extend(['y'])
		#norms.extend([(0, 12000)])
		#add.extend([self.ctrl.model.hzh, self.ctrl.err_h])
		#norms.extend([(0, 12000), (-12000, 12000)])

		return ks, add, norms, (len(ks)+len(add),)

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
		return self.ctrl.is_nan_err or self.ctrl.is_done or (not self.is_testing and self.ctrl.is_limit_err)

	def get_reward(self, action, action_prev):
		mode = self.reward_config.get('mode', 'standard')
		if action_prev is None:
			action_prev = action 
		if type(action) in [np.ndarray, list, np.array]:
			action, action_prev = action[0], action_prev[0]
		if mode == 'standard':
			if self.ctrl.no_correct or self.ctrl.manual_stab:
				k1, k2, k3 = self.reward_config.get('k1', 2), self.reward_config.get('k2', 1), self.reward_config.get('k3', 1)
				s = k1 + k2 + k3
				k1 /= s
				k2 /= s
				k3 /= s
				#if self.ctrl.model.state_dict['vartheta']*self.ctrl.model.dvartheta < 0:
				A = 0.5
				#else:
				#	A = 0.6
				r = A-k1*abs(self.ctrl.model.dvartheta)-k2*abs(self.ctrl.model.dvartheta_dt)-k3*abs(self.ctrl.model.dvartheta_dt_dt) #-self.ctrl.model.TAE #+0.6/(1+abs(action)) #
				#print(abs(self.ctrl.model.dvartheta), abs(self.ctrl.model.dvartheta_dt), abs(self.ctrl.model.dvartheta_dt_dt))
			elif self.ctrl.no_correct or self.ctrl.manual_ctrl:
				Ay = self.reward_config.get('Ay', 1)
				rmin, emax = 1e-3, 12000
				ky = (1/rmin-1)*1/emax
				ry = Ay/(1+ky*abs(self.ctrl.err_h))
				r = ry
			else:
				raise NotImplementedError
			return r
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
		self.ctrl.step(action)
		observation = self._get_obs()
		reward = self.get_reward(action, self.action_prev)
		self.action_prev = action
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