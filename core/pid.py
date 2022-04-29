import logging
from math import pi
import core
from core.model import Model
from tools.general import Storage

class ControllerPID:
    ''' Управляющий контроллер для управления средой только с помощью ПИД-регуляторов'''
    def __init__(
        self,
        tk:float=50,
        h_func=None,
        vartheta_func=None,
        use_ctrl:bool=False,
        use_storage=False,
        logging_path='ControllerPID.log'
    ):
        self.logger = logging.Logger("ControllerPID", logging.DEBUG)
        if logging_path:
            self.logger.addHandler(logging.FileHandler(logging_path))
        self.tk = tk
        self.h_func = h_func
        self.vartheta_func = vartheta_func
        self.use_ctrl = use_ctrl
        self.use_storage = use_storage

        self.logger.debug(f"Получены настройки:\n\ttk={self.tk}\n\th_func={self.h_func}\n\tvartheta_func={self.vartheta_func}\n\tuse_ctrl={self.use_ctrl}\n\tuse_storage={self.use_storage}")
        assert self.h_func is not None or self.vartheta_func is not None, "Не выставлены требуемые значения хотя бы одной из функций."
        assert not self.use_ctrl or (self.use_ctrl and self.h_func is not None), "Контроллер с СУ ПИД не может функционировать без функции требуемой высоты."
        if self.use_ctrl and self.vartheta_func is not None:
            self.logger.warning("Выставлен параметр use_ctrl=True, функция требуемого значения угла тангажа не будет учитываться.")
        self.storage = Storage()
        self.storage_backup = Storage()

        self.model = Model(use_PID_CS=self.use_ctrl, use_PID_SS=True)

        self.reset()


    def reset(self):
        self.logger.debug("Выполняю сброс контроллера.")
        self.model.initialize()
        self.storage_backup.storage = dict(self.storage.storage)
        self.storage.clear_all()


    def pre_step(self, state=None):
        pass


    def post_step(self, state=None):
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


    def step(self) -> bool:
        if self.use_ctrl:
            self.model.hzh = self.h_func(self.model.time)
        else:
            self.model.vartheta_zh = self.vartheta_func(self.model.time)
        self.pre_step()
        self.model.step()
        self.post_step()
        return self.model.time >= self.tk


    def get_info(self) -> dict:
        return self.model.state_dict
    


if __name__ == '__main__':
    ctrl_pid = ControllerPID(tk=60, h_func=lambda _: 5000, vartheta_func=None, use_ctrl=True)
    while ctrl_pid.step():
        print(ctrl_pid.get_info())
