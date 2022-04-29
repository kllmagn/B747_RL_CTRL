from cgitb import reset
from multiprocessing.sharedctypes import Value
from core.model import Model
from tools.general import *

from math import pi, sin
import random

import numpy as np

from enum import Enum

class CtrlMode(Enum):
    DIRECT_CONTROL = 0 # прямой режим управления
    ADD_PROC_CONTROL = 1 # режим относительной компенсирующей добавки
    ANG_VEL_CONTROL = 2 # управление по угловой скорости угла отклонения рулей
    ADD_DIRECT_CONTROL = 3 # режим прямой компенсирующей добавки

class AeroComponent(Enum):
    CXA = 0
    CYA = 1
    MZ = 2
    MZ_DELTAZ = 3

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
        action_max=17*pi/180,
        vartheta_max=10*pi/180,
        sample_time:float=None,
        use_limiter=False,
        random_init = False, # случайно назначить начальные значения высоты и угла тангажа при инициализации модели
        no_correct=False, # ПИД-СС (+ПИД-СУ при use_ctrl=True) в неуправляемом (автоматическом) режиме
        random_reset=True, # случайный сброс
        sim_init_state=True, # моделировать ПИД управление для создания начального состояния моделирования при случайном сбросе
        reset_state0:list=None,
        ctrl_mode=None, # режим управления
        reset_aero_err=None
        ):
        self.no_correct = no_correct
        self.use_ctrl = use_ctrl
        self.manual_ctrl = manual_ctrl
        self.manual_stab = manual_stab
        self.model = Model(use_PID_CS=self.use_ctrl and not self.manual_ctrl, use_PID_SS=not self.manual_stab)
        self.tk = tk # время окончания интегрирования
        self.sample_time = sample_time if sample_time else self.model.dt
        assert self.sample_time >= self.model.dt, "Шаг интегрирования не может превышать шаг взаимодействия."
        self.storage = Storage() # хранилище параметров
        self.storage_backup = Storage()
        self.use_storage = use_storage
        self.h_func = h_func # функция требуемой высоты от времени
        self.vartheta_func = vartheta_func
        self.random_reset = random_reset #self.h_func is None or self.vartheta_func is None
        self.random_init = random_init
        self.w_deltaz = 0
        self.action_max = action_max # максимальное значение по углу отклонения рулей
        self.vartheta_max = vartheta_max # максимальное значение угла тангажа
        self.deltaz_diff_int = Integrator()
        self.vth_err = Integrator()
        self.vth_err_abs = Integrator()
        self.deriv_dict = DerivativeDict()
        self.memory_dict = MemoryDict()
        self.deltaz_backup = 0
        self.state_backup = np.zeros(self.model.state.shape)
        self.use_limiter = use_limiter
        self.ctrl_mode = ctrl_mode
        self.sim_init_state = sim_init_state
        self.reset_state0 = reset_state0
        self.reset_vartheta_ref = None
        self.reset_aero_err = reset_aero_err

    def reset(self):
        # 0.01 Гц, 0.5 Гц | -10*pi/180<=A<=10*pi/180 | sin
        h0 = random.uniform(1000, 11000)
        if self.random_init:
            Vx = random.uniform(250, 265) #259.1667
            v0 = random.uniform(-10*pi/180, 10*pi/180)
            if self.sim_init_state:
                model = Model(use_PID_SS=True, use_PID_CS=False)
                v1 = random.uniform(-10*pi/180, 10*pi/180) if self.reset_state0 is None else 0
                state0 = [0, h0, 0, Vx, 0, 0, 0, 0, 0, 0, 0, v0, 0, 0, 0, 0] if self.reset_state0 is None else self.reset_state0
                model.set_initial(state0)
                model.initialize()
                model.vartheta_zh = v1
                #print('='*30+f'\nСимулирую полет на ПИД.\nvartheta0 = {v0*180/pi}\nvartheta1 = {v1*180/pi}\nstate0 = {state0}\n'+'='*30)
                while abs(v1-model.state_dict['vartheta']) > 0.01*pi/180:
                    model.step()
                #print('='*30+f'\nОкончание симуляции.\nvartheta1 = {model.state_dict["vartheta"]*180/pi}\nstate1 = {model.state}\n'+'='*30)
                self.model.set_initial(model.state)
            else:
                state0 = [0, h0, 0, Vx, 0, 0, 0, 0, 0, 0, 0, v0, 0, 0, 0, 0] if self.reset_state0 is None else self.reset_state0
                self.model.set_initial(state0)
        if self.random_reset:
            vartheta1 = random.uniform(-10*pi/180, 10*pi/180) if self.reset_vartheta_ref is None else self.reset_vartheta_ref
            self.vartheta_func = lambda _: vartheta1 
            for component in AeroComponent:
                err = random.uniform(-0.5, 0.5)
                self.apply_aero_err(component, err)
        if self.reset_aero_err is not None:
            for i in range(len(self.reset_aero_err)):
                self.apply_aero_err(AeroComponent(i), self.reset_aero_err[i])
        self.model.initialize()          
        self.memory_dict.input('dvedt', 0)
        self.storage_backup.storage = dict(self.storage.storage)
        self.storage.clear_all()
        self.deltaz_diff_int.reset()
        self.vth_err.reset()
        self.vth_err_abs.reset()
        #self.model.P = self.model.Pmax * random.uniform(0.01, 1)
        
        #self.model.step()
        #self.post_step()
        #self.model.initialize()

    def pre_step(self, state=None):
        pass

    def post_step(self, state=None):
        self.vth_err.input(self.err_vartheta)
        self.vth_err_abs.input(abs(self.err_vartheta))
        if self.use_storage:
            # используется режим записи состояния модели
            self.storage.record("t", self.model.time)
            self.storage.record('deltaz', self.model.deltaz_com*180/pi)
            self.storage.record('deltaz_ref', self.model.deltaz_ref*180/pi)
            self.storage.record('deltaz_real', self.model.deltaz_real*180/pi)
            self.storage.record('hzh', self.model.hzh)
            self.storage.record('vartheta_ref', self.vartheta_ref*180/pi)
            if state is None:
                state = self.model.state_dict
            angles = ['alpha', 'vartheta', 'psi', 'gamma']
            for k, v in state.items():
                if k in angles:
                    v *= 180/pi
                self.storage.record(k, v)

    def step(self, action:np.ndarray):
        '''
        Выполнить один шаг симуляции с заданным массивом управляющих параметров.
        '''
        # ================= Воздействие на модель ===================
        if not self.use_ctrl:
            self.model.vartheta_zh = self.vartheta_func(self.model.time)
        else:
            self.model.hzh = self.h_func(self.model.time)
            if self.model.use_PID_CS:
                if not self.no_correct:
                    self.model.PID_CS = (self.model._PID_initial*action)[:4] if self.model.use_PID_SS\
                        else self.model._PID_initial[:4]*action[:4]
            else:
                self.model.vartheta_zh = action[0]
        if self.model.use_PID_SS:
            if not self.no_correct:
                self.model.PID_SS = (self.model._PID_initial*action)[-4:] if self.model.use_PID_CS \
                    else self.model._PID_initial[-4:]*action[-4:]
        else:
            self.deltaz_diff_int.input(abs(action[-1]))
            if self.ctrl_mode is None or self.ctrl_mode == CtrlMode.DIRECT_CONTROL:
                self.model.deltaz = action[-1]
            elif self.ctrl_mode == CtrlMode.ADD_PROC_CONTROL:
                self.model.deltaz = np.clip([(1+action[-1])*self.model.deltaz_ref], [-17*pi/180], [17*pi/180])[0]
            elif self.ctrl_mode == CtrlMode.ADD_DIRECT_CONTROL:
                self.model.deltaz = np.clip([action[-1]+self.model.deltaz_ref], [-17*pi/180], [17*pi/180])[0]
            elif self.ctrl_mode == CtrlMode.ANG_VEL_CONTROL:
                self.model.deltaz = np.clip([self.model.deltaz+self.w_deltaz*self.sample_time], [-17*pi/180], [17*pi/180])[0]
            else:
                raise ValueError(f"Неподдерживаемый режим управления: {self.ctrl_mode}")
        # ===========================================================
        self.state_backup = self.model.state
        self.deltaz_backup = self.model.deltaz if self.manual_stab else self.model.deltaz_ref
        self.pre_step()
        self.model.step() # производим симуляцию модели на один шаг
        self.post_step()
        while ((round(round(self.model.time/self.model.dt) % round(self.sample_time/self.model.dt))) != 0):
            #if not self.model.use_PID_SS:
            #    self.model.deltaz = np.clip([self.model.deltaz+self.w_deltaz], [-17*pi/180], [17*pi/180])[0]
            self.pre_step()
            self.model.step()
            self.post_step()

    def apply_aero_err(self, component:AeroComponent, value:float):
        ind = component.value
        #print(f'applying {value} on {ind}')
        vec = self.model.aero_err
        vec[ind] = value
        self.model.aero_err = vec

    @property
    def deltaz(self):
        return self.model.deltaz_com #deltaz if not self.model.use_PID_SS else self.model.deltaz_ref

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
        '''Ошибка между требуемым и фактическим углами тангажа.'''
        return self.vartheta_ref-self.model.state_dict['vartheta']

    @property
    def err_vartheta_rel(self) -> float:
        '''Относительная ошибка между требуемым и фактическим углами тангажа.'''
        return self.err_vartheta if self.vartheta_ref == 0 else self.err_vartheta/self.vartheta_ref

    @property
    def err_h(self) -> float:
        '''Ошибка между требуемым и фактическим значениями высоты.'''
        return self.model.hzh-self.model.state_dict['y']
        
    @property
    def is_limit_err(self) -> bool:
        '''Произошла ли ошибка вследствие превышения ограничений по углам.'''
        return self.use_limiter and (abs(self.model.state_dict['vartheta']) > 5*pi/180+self.vartheta_max or self.model.deltaz > self.action_max)

    @property
    def is_nan_err(self) -> bool:
        return np.isnan(np.sum(self.model.state))

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

    def stepinfo_SS(self, use_backup=False) -> dict:
        "Характеристики ПП СС"
        storage = self.storage_backup if use_backup else self.storage
        if not self.use_storage or (self.use_storage and ('vartheta' not in storage.storage or 't' not in storage.storage)):
            raise ValueError('Вычисление хар-к ПП СС недоступно: ошибка хранилища.')
        return calc_stepinfo(storage.storage['vartheta'], storage.storage['vartheta_ref'][-1], ts=storage.storage['t'])

    def stepinfo_CS(self, use_backup=False) -> dict:
        "Характеристики ПП СУ"
        storage = self.storage_backup if use_backup else self.storage
        if not self.use_storage or (self.use_storage and ('y' not in storage.storage or 't' not in storage.storage)):
            raise ValueError('Вычисление хар-к ПП СУ недоступно: ошибка хранилища.')
        return calc_stepinfo(storage.storage['y'], storage.storage['hzh'][-1], ts=storage.storage['t'])

def main():
    vartheta_func = lambda t: 0.1
    h_func = lambda t: 12000
    ctrl = Controller(h_func=h_func, vartheta_func=vartheta_func, use_ctrl=True, manual_ctrl=False, manual_stab=False, no_correct=True)
    while not ctrl.is_done:
        ctrl.step([])

if __name__ == '__main__':
	main()