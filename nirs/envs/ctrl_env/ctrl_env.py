import numpy as np
import gym
from math import atan, exp, pi, log, tan
from gym import spaces
from pandas import array

from .tools import calc_err
from .ctrl import Controller, CtrlMode

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
		self.deltaz_ref_back = self.ctrl.model.deltaz_ref

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
			acts_low.extend([-self.ctrl.action_max])
			acts_high.extend([self.ctrl.action_max])
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
		#norms.extend([(-self.ctrl.action_max, self.ctrl.action_max)])
		#norms.extend([(-self.ctrl.action_max, self.ctrl.action_max)])

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

	def get_reward(self, action, action_prev, deltaz_ref_back):
		mode = self.reward_config.get('mode', 'standard')
		if action_prev is None:
			action_prev = action 
		if type(action) in [np.ndarray, list, np.array]:
			action, action_prev = action[0], action_prev[0]
		if mode == 'standard':
			if self.ctrl.no_correct or self.ctrl.manual_stab:
				A = 1.0
				k1, k2, k3 = self.reward_config.get('k1', 2), self.reward_config.get('k2', 2), self.reward_config.get('k3', 1)
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0:
					# перерегулирование с противоположной стороны
					self.dv_max = max(self.dv_max, self.ctrl.model.dvartheta) if self.ctrl.model.dvartheta > 0 else min(self.dv_max, self.ctrl.model.dvartheta)
					#k1, k2, k3 = 1, 0, 0
					#print(self.dv_max/self.ctrl.vartheta_ref*100)
				s = k1 + k2 + k3
				k1 /= s
				k2 /= s
				k3 /= s
				k2_0 = 1/(1+5*abs(self.ctrl.model.dvartheta)/abs(2*self.ctrl.vartheta_ref))
				#r = A-10*pi/180*(k1*abs(self.ctrl.model.dvartheta/self.ctrl.vartheta_ref)+k2*abs(self.ctrl.model.dvartheta_dt/self.ctrl.vartheta_ref)+k3*abs(self.ctrl.model.dvartheta_dt_dt/self.ctrl.vartheta_ref)) #-abs(self.dv_max)) #+1/(1+0.01*abs(self.dv_max)/abs(self.ctrl.vartheta_ref)) #-self.ctrl.model.TAE #+0.6/(1+abs(action)) #
				dw = 1/100*2*self.ctrl.vartheta_max/self.ctrl.model.dvartheta*self.ctrl.model.dvartheta_dt
				#r = exp(-abs(self.ctrl.model.dvartheta/(self.ctrl.model.vartheta_ref))) #2*self.ctrl.vartheta_max-abs(self.ctrl.model.dvartheta)-abs(dw) #-0.1*abs(self.dv_max/self.ctrl.vartheta_ref)
				r = exp(-(k1*abs(self.ctrl.model.dvartheta)+k2*k2_0*abs(self.ctrl.model.dvartheta_dt)+k3*abs(self.ctrl.model.dvartheta_dt_dt))/abs(2*self.ctrl.vartheta_ref)) #-abs(self.dv_max)
				if self.ctrl.vartheta_ref*self.ctrl.model.dvartheta < 0:
					r *= 0.2
				k = abs(self.ctrl.model.dvartheta/(2*self.ctrl.vartheta_max))
				rf = -k*(abs(action-self.ctrl.model.deltaz_ref))/(34*pi/180) if self.ctrl.ctrl_mode == CtrlMode.DIRECT_CONTROL else 0
				r = r + rf
				'''
				if self.ctrl.model.time == self.ctrl.tk:
					def transfer_quality():
						info = self.ctrl.stepinfo_SS(use_backup=False)
						time, overshoot = info['settling_time'], info['overshoot']
						if time is None or overshoot is None:
							return np.inf
						else:
							return time*abs(overshoot)
					r -= transfer_quality()*0.1
				'''
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
			return rt
		elif mode == 'funclike':
			tp = 3
			y = self.ctrl.model.vartheta_ref
			b = y/(pi/2)
			a = tan(0.95*y/b)/tp
			y_exp = b*atan(a*self.ctrl.model.time)
			w_exp = b/(1+a**2*self.ctrl.model.time**2)
			r = exp((-abs(self.ctrl.model.state_dict['vartheta']-y_exp)-abs(self.ctrl.model.state_dict['wz']-w_exp))/(20*pi/180))
			return r
		else:
			return 0.5/(self.ctrl.calc_CS_err()+1) + 0.5/(self.ctrl.calc_SS_err()+1)

	def step(self, action):
		self.ctrl.step(action)
		observation = self._get_obs()
		reward = self.get_reward(action, self.action_prev, self.deltaz_ref_back)
		self.action_prev = action
		self.deltaz_ref_back = self.ctrl.model.deltaz_ref
		done = self.is_done()
		info = {} #'storage': self.ctrl.storage}
		return observation, reward, done, info

	def reset(self):
		self.dv_max = 0
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