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

		ks.extend(['alpha', 'vartheta', 'wz'])
		norms.extend([(-pi, pi), (-pi, pi), (-pi, pi)])
		add.extend([self.ctrl.err_vartheta, self.ctrl.vartheta_ref]) #, self.ctrl.model.deltaz_real, self.ctrl.model.deltaz])
		norms.extend([(-pi, pi), (-pi, pi)]) #, (-pi, pi), (-pi, pi)])

		#ks.extend(['y'])
		#norms.extend([(0, 80000)])
		#add.extend([self.ctrl.model.hzh, self.ctrl.err_h])
		#norms.extend([(0, 80000), (-80000, 80000)])

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
		return self.ctrl.is_nan_err or self.ctrl.is_done or (not self.is_testing and self.ctrl.is_limit_err)

	def get_reward(self, action):
		mode = self.reward_config.get('mode', 'standard')
		if mode == 'standard':
			if self.ctrl.no_correct or self.ctrl.manual_stab:
				rmin = 0.07
				v_max, vint_max, w_max, deltaz_max = 20*pi/180, 1000, 0.2, 50*pi/180
				calc_k = lambda max_val: -log(rmin)/max_val # e^(-k*max_val) = rmin => -k*max_val = log(rmin) => k = -log(rmin)/max_val
				Av = self.reward_config.get('Av', 1)
				Avint = self.reward_config.get('Avint', 0)
				Aw = self.reward_config.get('Aw', 0)
				Adeltaz = self.reward_config.get('Adeltaz', 0)
				kv = self.reward_config.get('kv', calc_k(v_max)) #1/(20*pi/180)) #180/pi)
				kvint = self.reward_config.get('kvint', calc_k(vint_max))
				kw = self.reward_config.get('kw', calc_k(w_max)) #0) #0.1/0.1)
				kdeltaz = self.reward_config.get('kdeltaz', calc_k(deltaz_max)) #0) #1/(34*pi/180))
				dvartheta = self.reward_config.get('dvartheta', 20*pi/180)
				use_limit_punisher = True
				#rv = Av/(1+kv*abs(self.ctrl.err_vartheta))
				#rvint = Avint/(1+10*self.ctrl.vth_err.output()) #kv*abs(self.ctrl.err_vartheta))
				#rdeltaz = Adeltaz/(1+kdeltaz*(abs(self.ctrl.err_vartheta)/dvartheta)*abs(self.ctrl.deltaz-self.ctrl.model.deltaz_ref))
				#rw = Aw/(1+kw*abs(self.ctrl.model.state_dict['wz'])*k1*abs(self.ctrl.err_vartheta)/dvartheta)
				#rl = -2 if (use_limit_punisher and self.ctrl.is_limit_err) else 0
				rv = Av*exp(-kv*abs(self.ctrl.err_vartheta))
				rw = Aw*exp(-kw*abs(self.ctrl.model.state_dict['wz']))
				rvint = Avint*exp(kvint*abs(self.ctrl.vth_err.output()))
				rdeltaz = Adeltaz*exp(-kdeltaz*abs(self.ctrl.deltaz-self.ctrl.model.deltaz_ref)) #rv+rw+rvint+rdeltaz+rl
				r = rv+rw+rvint+rdeltaz
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