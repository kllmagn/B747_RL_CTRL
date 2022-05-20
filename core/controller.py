import logging
from pathlib import Path
from tools.general import *

from .model import Model

from math import pi, sin
import random
from enum import Enum
from typing import Callable, Union

import numpy as np

class CtrlType(Enum):
    FULL_AUTO = 0 # СУ ПИД + СС ПИД
    AUTO = 1 # СС ПИД
    SEMI_MANUAL = 2 # СУ ПИД + СС НС
    MANUAL = 3 # СС НС

class CtrlMode(Enum):
    DIRECT_CONTROL = 0 # прямой режим управления
    ADD_PROC_CONTROL = 1 # режим относительной компенсирующей добавки
    ANG_VEL_CONTROL = 2 # управление по производной управляющего сигнала
    ADD_DIRECT_CONTROL = 3 # режим прямой компенсирующей добавки

class ResetRefMode(Enum):
    CONST = 0 # Метод постоянного угла тангажа 
    OSCILLATING = 1 # Колебательная зависимость угла тангажа
    HYBRID = 2 # Гибридный метод

class DisturbanceMode(Enum):
    AERO_DISTURBANCE = 0 # Погрешности а/д коэффициентов в виде шума

class AeroComponent(Enum):
    CXA = 0
    CYA = 1
    MZ = 2
    MZ_DELTAZ = 3


class Controller:
    '''
    Управляющий контроллер MATLAB модели.\n

    Составы контура:\n
    \tСУ ПИД + СС ПИД (FULL_AUTO);\n
    \tСУ ПИД + СС НС (SEMI_MANUAL);\n
    \tСС ПИД (AUTO);\n
    \tСС НС (MANUAL);\n\n

    Режимы управления СС НС\n
    \tСС НС - Прямой режим управления (ПУ) (DIRECT_CONTROL);\n
    \tСС НС - Режим относительной компенсирующей добавки (ОКД) (ADD_PROC_CONTROL);\n
    \tСС НС - Управление по производной управляющего сигнала (ADD_VEL_CONTROL);\n
    \tСС НС - Режим прямой компенсирующей добавки (ПКД) (ADD_DIRECT_CONTROL).\n\n

    Методы инициализации требуемых значений:\n
    \tМетод постоянного угла тангажа (CONST);\n
    \tКолебательная зависимость угла тангажа (OSCILLATING);\n
    \tГибридный метод (HYBRID).\n\n

    Методы инициализации начального состояния моделирования:\n
    \tКлассическая прямая инициализация (DIRECT);\n
    \tМетод косвенной инициализации (INDIRECT);\n\n

    Методы инициализации возмущений:\n
    \tПогрешности а/д коэффициентов (AERO).\n\n
    '''

    def __init__(
        self,
        ctrl_type:CtrlType, # состав контура
        ctrl_mode:CtrlMode, # режим управления
        reset_ref_mode:ResetRefMode=None,
        disturbance_mode:DisturbanceMode=None,
        tk:float=60, # время окончания интегрирования
        sample_time:float=None, # шаг взаимодействия
        h_func:Callable[[float], float]=None, # функция требуемой высоты полета от времени
        vartheta_func:Callable[[float], float]=None, # функция требуемого угла тангажа от времени
        use_storage:bool=False, # использовать хранилище данных для их последующего анализа
        action_max:float=17*pi/180, # максимальное значение параметра вектора управления
        vartheta_max:float=10*pi/180,
        use_limiter:bool=False,
        logging_path:Path=None,
        aero_err:np.ndarray=None
        ):
        '''Создать объект контроллера.'''

        self.logger = logging.Logger(f"[Controller]", logging.DEBUG)
        if logging_path:
            fh = logging.FileHandler(logging_path, 'w', encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

        self.ctrl_type = ctrl_type
        self.ctrl_mode = ctrl_mode
        self.reset_ref_mode = reset_ref_mode
        self.disturbance_mode = disturbance_mode
        self.aero_err = aero_err
        self.logger.debug(f"Инициализация контроллера: ctrl_type = {self.ctrl_type}, ctrl_mode = {self.ctrl_mode}, reset_ref_mode = {self.reset_ref_mode}")

        assert self.ctrl_mode is not None or (self.ctrl_mode is None and self.ctrl_type in [CtrlType.AUTO, CtrlType.FULL_AUTO]), 'Режим управления не может быть None при наличии СС НС.'
        if self.ctrl_mode is not None and self.ctrl_type in [CtrlType.AUTO, CtrlType.FULL_AUTO]:
            self.logger.warn(f"Наличие режима управления при отсутствии СС НС [{self.ctrl_mode}]: игнорирование режима")
        
        self._init_model()

        self.sample_time = sample_time if sample_time else self.model.dt # если шаг взаимодействия указан, выставляем его; иначе он равен шагу интегрирования
        assert self.sample_time >= self.model.dt, "Шаг интегрирования не может превышать шаг взаимодействия."
        
        self.no_correct = True # коррекции не производится
        self.tk = tk # время окончания интегрирования
        
        self.h_func = h_func # функция требуемой высоты от времени
        self.vartheta_func = vartheta_func # функция требуемого угла тангажа от времени
        self.action_max = action_max # максимальное значение параметра вектора управления
        self.vartheta_max = vartheta_max # максимальное значение угла тангажа
        self.state_backup = np.zeros(self.model.state.shape)
        self.use_limiter = use_limiter

        self.use_storage = use_storage # следует ли использовать хранилище
        self.storage = Storage() # хранилище параметров
        self.storage_backup = Storage() # резервное хранилище параметров


    def _init_model(self) -> None:
        self.use_ctrl = self.ctrl_type in [CtrlType.SEMI_MANUAL, CtrlType.FULL_AUTO] # если СУ ПИД + СС НС или СУ ПИД + СС ПИД
        self.manual_stab = self.ctrl_type in [CtrlType.MANUAL, CtrlType.SEMI_MANUAL] # если СУ ПИД + СС НС или СС НС
        self.model = Model(use_PID_CS=self.use_ctrl, use_PID_SS=not self.manual_stab) # мат. модель Matlab


    def reset(self, state0:np.ndarray=None):
        '''Произвести сброс контроллера.'''
        self.logger.debug(f"[reset] Сброс контроллера")
        
        if state0 is not None and len(state0) > 0:
            self.logger.debug(f"[reset] state0 = {state0}")
            assert state0.shape == self.model.state.shape, "Размерности заданного вектора состояния state0 и вектора состояния модели не совпадают."
            self.model.set_initial(state0)
            assert self.reset_ref_mode is None, "Попытка случайного сброса при наличии начального вектора состояния."

        if self.reset_ref_mode is not None:
            assert self.ctrl_type in [CtrlType.SEMI_MANUAL, CtrlType.MANUAL], "Случайный сброс не поддерживается при отсутствии СС НС в контуре."
            # выполнить сброс в какой-либо форме
            self.logger.debug(f"[reset] Выполняется случайный сброс")
            h0 = random.uniform(1000, 11000)
            Vx = random.uniform(100, 265)
            Vy = random.uniform(-20, 20)
            wz0 = random.uniform(-0.001, 0.001)
            vartheta0 = random.uniform(-self.vartheta_max, self.vartheta_max)
            if self.reset_ref_mode == ResetRefMode.CONST:
                vartheta_ref = random.uniform(-self.vartheta_max, self.vartheta_max)
                self.vartheta_func = lambda _: vartheta_ref
            elif self.reset_ref_mode == ResetRefMode.OSCILLATING:
                # 0.01 Гц, 0.5 Гц | -10*pi/180<=A<=10*pi/180 | sin
                A1 = random.uniform(0, self.vartheta_max)
                A2 = random.uniform(0, self.vartheta_max-A1)
                A3 = random.uniform(0, self.vartheta_max-A1-A2)
                f1 = random.uniform(0.01, 0.5)
                f2 = random.uniform(0.01, 0.5)
                f3 = random.uniform(0.01, 0.5)
                self.vartheta_func = lambda t: A1*sin(2*pi*f1*t)+A2*sin(2*pi*f2*t)+A3*sin(2*pi*f3*t)
                # добавить ограничение на значение
            elif self.reset_ref_mode == ResetRefMode.HYBRID:
                assert self.ctrl_type in [CtrlType.MANUAL, CtrlType.SEMI_MANUAL], "Гибридный режим сброса поддерживается только для СС НС в составе контура."
                use_ctrl = random.choice([True, False])
                if use_ctrl:
                    self.ctrl_type = CtrlType.SEMI_MANUAL
                    h1 = h0 + random.uniform(-1000, 1000)
                    self.h_func = lambda _: h1
                else:
                    self.ctrl_type = CtrlType.MANUAL
                    vartheta_ref = random.uniform(-self.vartheta_max, self.vartheta_max)
                    self.vartheta_func = lambda _: vartheta_ref
                self._init_model()
            self.model.set_initial([0, h0, Vx, Vy, vartheta0, wz0])

        if self.disturbance_mode == DisturbanceMode.AERO_DISTURBANCE:
            self.logger.debug("[_pre_step] Выставляю случайную ошибку в а/д коэффициентах")
            if self.aero_err is None:
                self.model.aero_err =\
                    np.array([
                        np.random.normal(-0.1, 0.5, size=None),
                        np.random.normal(0.1, 0.5, size=None),
                        np.random.normal(-0.1, 0.5, size=None),
                        np.random.normal(-0.1, 0.5, size=None),
                        np.random.normal(0.1, 0.5, size=None),
                    ])
            else:
                self.model.aero_err = self.aero_err

        if self.use_storage:
            self.logger.debug("[reset] Произвожу резервное копирование хранилища")
            # если используется хранилище, производим резервное копирование данных во время сброса
            self.storage_backup.storage = dict(self.storage.storage)
            self.storage.clear_all()

        self.model.initialize()


    def _pre_step(self, action:np.ndarray, state=None):
        '''Выполнить действия перед шагом интегрирования.'''
        pass


    def _post_step(self, action:np.ndarray, state=None):
        '''Выполнить действия после шага интегрирования.'''
        if self.use_storage:
            self.logger.debug("[_post_step] Запись параметров в хранилище")
            # используется режим записи состояния модели
            self.storage.record("t", self.model.time)
            self.storage.record('U_com', self.model.deltaz_com)
            self.storage.record('U_PID', self.model.deltaz_ref)
            self.storage.record('deltaz', self.model.deltaz_real*180/pi)
            self.storage.record('hzh', self.model.hzh)
            self.storage.record('vartheta_ref', self.vartheta_ref*180/pi)
            if action is not None:
                self.storage.record('U_RL', action[0])
            if state is None:
                state = self.model.state_dict
            angles = ['vartheta']
            for k, v in state.items():
                if k in angles:
                    v *= 180/pi
                self.storage.record(k, v)


    def step(self, action:Union[np.ndarray, None]=None):
        '''Выполнить один шаг симуляции с заданным массивом управляющих параметров.'''
        # ================= Воздействие на модель ===================
        if not self.use_ctrl: # если СУ ПИД НЕ в составе контура
            # выставляем требуемый угол тангажа в соответствии с функцией
            self.model.vartheta_zh = self.vartheta_func(self.model.time)
        else: # если СУ ПИД в составе контура
            # выставляем требуемую высоту в соответствии с функцией
            self.model.hzh = self.h_func(self.model.time)
            self.model.vartheta_zh = action[0]
        if not self.model.use_PID_SS: # если СС ПИД НЕ в составе контура
            if self.ctrl_mode is None or self.ctrl_mode == CtrlMode.DIRECT_CONTROL:
                self.model.deltaz = action[-1]
            elif self.ctrl_mode == CtrlMode.ADD_PROC_CONTROL:
                self.model.deltaz = np.clip([(1+action[-1])*self.model.deltaz_ref], [-17*pi/180], [17*pi/180])[0]
            elif self.ctrl_mode == CtrlMode.ADD_DIRECT_CONTROL:
                self.model.deltaz = np.clip([action[-1]+self.model.deltaz_ref], [-17*pi/180], [17*pi/180])[0]
            elif self.ctrl_mode == CtrlMode.ANG_VEL_CONTROL:
                self.model.deltaz = np.clip([self.model.deltaz+action[-1]*self.sample_time], [-17*pi/180], [17*pi/180])[0]
            else:
                raise ValueError(f"Неподдерживаемый режим управления: {self.ctrl_mode}")
            self.logger.debug(f"[step] Симуляция шага интегрирования [{self.ctrl_mode}]: deltaz = {self.model.deltaz}")
        else:
            self.logger.debug(f"[step] Симуляция шага интегрирования: режим СС ПИД")
        # ===========================================================
        # производим резервное копирование вектора состояния мат. модели
        self.state_backup = self.model.state
        self._pre_step(action) # выполняем операции перед шагом интегрирования
        self.model.step() # производим симуляцию модели на один шаг
        self._post_step(action) # выполняем операции после шага интегрирования
        # выполняем интегрирование до тех пор пока не будет достигнут шаг взаимодействия
        while ((round(round(self.model.time/self.model.dt) % round(self.sample_time/self.model.dt))) != 0):
            self._pre_step(action)
            self.model.step()
            self._post_step(action)


    @property
    def vartheta_ref(self):
        '''Требуемое значение угла тангажа.'''
        return self.model.vartheta_ref if self.model.use_PID_CS else self.model.vartheta_zh
        

    @property
    def dstate(self) -> np.ndarray:
        '''Разность вектора состояния модели.'''
        return self.model.state-self.state_backup


    @property
    def dstate_dict(self) -> dict:
        '''Именованная разность вектора состояния модели.'''
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
        '''Произошла ли ошибка внутри мат. модели Matlab.'''
        return np.isnan(np.sum(self.model.state))


    @property
    def is_done(self) -> bool:
        '''Достигнуто ли окончание моделирования.'''
        return self.model.time >= self.tk


    def calc_SS_err(self) -> float:
        '''Вычисление ошибки СС ЛА.'''
        vartheta = self.model.state_dict['vartheta']
        return calc_err(vartheta, self.vartheta_ref)


    def calc_CS_err(self) -> float:
        '''Вычисление ошибки СУ ЛА.'''
        h = self.model.state_dict['y']
        return calc_err(h, self.model.hzh)


    def stepinfo_SS(self, use_backup=False) -> dict:
        "Характеристики ПП СС."
        storage = self.storage_backup if use_backup else self.storage
        if not self.use_storage or (self.use_storage and ('vartheta' not in storage.storage or 't' not in storage.storage)):
            raise ValueError('Вычисление хар-к ПП СС недоступно: ошибка хранилища.')
        return calc_stepinfo(storage.storage['vartheta'], storage.storage['vartheta_ref'][-1], ts=storage.storage['t'])


    def stepinfo_CS(self, use_backup=False) -> dict:
        "Характеристики ПП СУ."
        storage = self.storage_backup if use_backup else self.storage
        if not self.use_storage or (self.use_storage and ('y' not in storage.storage or 't' not in storage.storage)):
            raise ValueError('Вычисление хар-к ПП СУ недоступно: ошибка хранилища.')
        return calc_stepinfo(storage.storage['y'], storage.storage['hzh'][-1], ts=storage.storage['t'])


def main():
    ctrl = Controller(ctrl_type=CtrlType.FULL_AUTO, ctrl_mode=None)
    while not ctrl.is_done:
        ctrl.step([])

if __name__ == '__main__':
	main()