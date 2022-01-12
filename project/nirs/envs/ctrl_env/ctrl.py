from .model import Model
from .tools import *

from math import pi
import random

import numpy as np

class Controller:
    '''Управляющий контроллер MATLAB модели.'''

    def __init__(
        self,
        tk:float=50, # время окончания интегрирования
        h_func=None, # функция требуемой высоты полета от времени
        vartheta_func=None, # функция требуемого угла тангажа от времени
        use_ctrl:bool=False, # использовать СУ
        manual_ctrl:bool=True, # использовать ручное управление (СУ)
        manual_stab:bool=True, # использовать ручное управление (СС)
        use_storage=False,
        delta_max=15*pi/180,
        vartheta_max=10*pi/180,
        sample_step:float=None,
        use_limiter=True
        ):
        self.model = Model(use_PID_CS=use_ctrl and not manual_ctrl, use_PID_SS=not manual_stab)
        self.tk = tk # время окончания интегрирования
        self.sample_step = sample_step if sample_step else self.model.dt
        assert self.sample_step >= self.model.dt, "Шаг интегрирования не может превышать шаг взаимодействия."
        self.storage = Storage() # хранилище параметров
        self.use_storage = use_storage
        self.h_func = h_func # функция требуемой высоты от времени
        self.vartheta_func = vartheta_func
        self.random_reset = self.h_func is None or self.vartheta_func is None
        self.delta_max = delta_max # максимальное значение по углу отклонения рулей
        self.vartheta_max = vartheta_max # максимальное значение угла тангажа
        self.use_ctrl = use_ctrl
        self.manual_ctrl = manual_ctrl
        self.manual_stab = manual_stab
        self.vth_err = Integrator()
        self.deriv_dict = DerivativeDict()
        self.memory_dict = MemoryDict()
        self.state_backup = np.zeros(self.model.state.shape)
        self.use_limiter = use_limiter

    def reset(self):
        if self.random_reset:
            h_zh = random.uniform(10800, 11300)
            vartheta_zh = random.uniform(-10*pi/180, 10*pi/180)
            self.h_func = lambda t: h_zh
            self.vartheta_func = lambda t: vartheta_zh
            #print('Устанавливаю vartheta_zh =', vartheta_zh*180/pi)
        self.model.initialize()
        self.storage.clear_all()
        self.vth_err.reset()

    def post_step(self):
        self.vth_err.input(self.err_vartheta)
        self.deriv_dict.input('dvedt', self.err_vartheta, self.model.dt)
        if self.use_storage:
            # используется режим записи состояния модели
            self.storage.record("t", self.model.time)
            self.storage.record('deltaz', self.model.deltaz)
            self.storage.record('hzh', self.model.hzh)
            self.storage.record('vartheta_ref', self.vartheta_ref)
            state = self.model.state_dict
            for k, v in state.items():
                self.storage.record(k, v)

    def step(self, action:np.ndarray):
        '''
        Выполнить один шаг симуляции с заданным массивом управляющих параметров.
        '''
        # выставляем значение текущей треубемой высоты в соответствии со значением заданной функции
        self.model.hzh = self.h_func(self.model.time)
        if self.model.use_PID_CS:
            self.model.PID_CS = (self.model._PID_initial*action)[:4] if self.model.use_PID_SS\
                else self.model._PID_initial[:4]*action[:4]
        elif not self.use_ctrl:
            self.model.vartheta_zh = self.vartheta_func(self.model.time)
        else:
            self.model.vartheta_zh = action[0]
        if self.model.use_PID_SS:
            self.model.PID_SS = (self.model._PID_initial*action)[-4:] if (self.model.use_PID_CS or not self.use_ctrl) \
                else self.model._PID_initial[-4:]*action[-4:]
        else:
            self.model.deltaz = action[-1]
        self.state_backup = self.model.state
        self.model.step() # производим симуляцию модели на один шаг
        self.post_step()
        while ((round(round(self.model.time/self.model.dt) % round(self.sample_step/self.model.dt))) != 0):
            self.model.step()
            self.post_step()

    @property
    def vartheta_ref(self):
        return self.model.vartheta_ref if self.model.use_PID_CS else self.model.vartheta_zh
        
    @property
    def dstate(self) -> np.ndarray:
        '''Разность вектора состояния модели.'''
        return self.model.state-self.state_backup

    @property
    def dstate_dict(self) -> dict:
        dst = self.dstate
        return dict((self.model.labels[i], dst[i]) for i in range(len(self.model.labels)))

    @property
    def err_vartheta(self) -> float:
        '''Ошибка между требуемым (СУ) и фактическим углами тангажа.'''
        return self.vartheta_ref-self.model.state_dict['vartheta']

    @property
    def is_limit_err(self) -> bool:
        '''Произошла ли ошибка вследствие превышения ограничений по углам.'''
        return self.use_limiter and (abs(self.model.state_dict['vartheta']) > 5*pi/180+self.vartheta_max or self.model.deltaz > self.delta_max)

    @property
    def is_done(self) -> bool:
        return self.model.time >= self.tk

    def calc_SS_err(self) -> float:
        '''Вычисление ошибки СС ЛА'''
        vartheta = self.model.state_dict['vartheta']
        return calc_err(vartheta, self.vartheta_ref)

    def calc_CS_err(self) -> float:
        '''Вычисление ошибки СУ ЛА'''
        h = self.model.state_dict['y']
        return calc_err(h, self.model.hzh)

    def stepinfo_SS(self) -> dict:
        "Характеристики ПП СС"
        if not self.use_storage or (self.use_storage and ('vartheta' not in self.storage.storage or 't' not in self.storage.storage)):
            raise ValueError('Вычисление хар-к ПП СС недоступно: ошибка хранилища.')
        return calc_stepinfo(self.storage.storage['vartheta'], self.vartheta_ref, ts=self.storage.storage['t'])

    def stepinfo_CS(self) -> dict:
        "Характеристики ПП СУ"
        if not self.use_storage or (self.use_storage and ('y' not in self.storage.storage or 't' not in self.storage.storage)):
            raise ValueError('Вычисление хар-к ПП СУ недоступно: ошибка хранилища.')
        return calc_stepinfo(self.storage.storage['y'], self.model.hzh, ts=self.storage.storage['t'])

def main():
	pass

if __name__ == '__main__':
	main()