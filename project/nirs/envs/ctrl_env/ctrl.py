from .model import Model
from .tools import calc_stepinfo, calc_err, Storage, Integrator

from math import pi
import random

import numpy as np

class Controller:
    '''Управляющий контроллер MATLAB модели.'''

    def __init__(
        self,
        tk:float=50, # время окончания интегрирования
        h_func=None, # функция требуемой высоты полета от времени
        manual_control:bool=True, # использовать ручное управление (только с поддержкой ПИД-регулятора СУ)
        use_storage=False,
        delta_max=15*pi/180,
        vartheta_max=10*pi/180,
        ):
        self.model = Model(use_PID_CS=True, use_PID_SS=not manual_control)
        self.tk = tk # время окончания интегрирования
        self.storage = Storage() # хранилище параметров
        self.use_storage = use_storage
        self.h_func = h_func # функция требуемой высоты от времени
        self.use_random_h = self.h_func is None
        self.delta_max = delta_max # максимальное значение по углу отклонения рулей
        self.vartheta_max = vartheta_max # максимальное значение угла тангажа
        self.manual_control = manual_control
        self.vth_err = Integrator()
        self.state_backup = np.zeros(self.model.state.shape)

    def reset(self):
        if self.use_random_h:
            h0 = random.uniform(10200, 12000)
            self.h_func = lambda t: h0
        self.model.initialize()
        self.storage.clear_all()
        self.vth_err.reset()

    def step(self, action:np.ndarray):
        '''
        Выполнить один шаг симуляции с заданным массивом управляющих параметров.
        '''
        # выставляем значение текущей треубемой высоты в соответствии со значением заданной функции
        self.model.hzh = self.h_func(self.model.time)
        if self.model.use_PID_SS and self.model.use_PID_CS:
            # используется поддержка обоих ПИД-регуляторов, изменяем их коэф-ты в соответствии с управлением
            action = self.model._PID_initial * action
            self.model.PID_CS = action[:int(action.size/2)]
            self.model.PID_SS = action[int(action.size/2):]
        else:
            # используется поддержка ПИД-регулятора СУ, выставляем значения угла отклонения рулей
            '''
            wz = action[0]
            deltaz = self.model.deltaz
            deltaz += wz*self.model.dt
            deltaz = min(15*pi/180, max(-15*pi/180, deltaz))
            '''
            self.model.deltaz = action[0]
        self.state_backup = self.model.state
        self.model.step() # производим симуляцию модели на один шаг
        self.vth_err.input(self.err_vartheta)
        if self.use_storage:
            # используется режим записи вектора состояния модели
            self.storage.record("t", self.model.time)
            self.storage.record('deltaz', self.model.deltaz)
            self.storage.record('hzh', self.model.hzh)
            self.storage.record('vartheta_ref', self.model.vartheta_ref)
            state = self.model.state_dict
            for k, v in state.items():
                self.storage.record(k, v)

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
        return self.model.vartheta_ref-self.model.state_dict['vartheta']

    @property
    def is_limit_err(self) -> bool:
        '''Произошла ли ошибка вследствие превышения ограничений по углам.'''
        return self.model.state_dict['vartheta']*0.9 > self.vartheta_max or self.model.deltaz > self.delta_max

    @property
    def is_done(self) -> bool:
        return self.model.time >= self.tk

    def calc_SS_err(self) -> float:
        '''Вычисление ошибки СС ЛА'''
        vartheta = self.model.state_dict['vartheta']
        return calc_err(vartheta, self.model.vartheta_ref)

    def calc_CS_err(self) -> float:
        '''Вычисление ошибки СУ ЛА'''
        h = self.model.state_dict['y']
        return calc_err(h, self.model.hzh)

    def stepinfo_SS(self) -> dict:
        "Характеристики ПП СС"
        if not self.use_storage or (self.use_storage and ('vartheta' not in self.storage.storage or 't' not in self.storage.storage)):
            raise ValueError('Вычисление хар-к ПП СС недоступно: ошибка хранилища.')
        return calc_stepinfo(self.storage.storage['vartheta'], self.model.hzh, ts=self.storage.storage['t'])

    def stepinfo_CS(self) -> dict:
        "Характеристики ПП СУ"
        if not self.use_storage or (self.use_storage and ('y' not in self.storage.storage or 't' not in self.storage.storage)):
            raise ValueError('Вычисление хар-к ПП СУ недоступно: ошибка хранилища.')
        return calc_stepinfo(self.storage.storage['y'], self.hzh, ts=self.storage.storage['t'])

def main():
	pass

if __name__ == '__main__':
	main()