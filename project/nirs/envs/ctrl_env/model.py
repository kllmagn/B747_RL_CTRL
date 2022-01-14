import ctypes
import os
import platform
import random
from math import pi
import pathlib
from shutil import copyfile
import tempfile
import pathlib
import weakref

import uuid
import numpy as np

from .rtwtypes import real_T, boolean_T

class Model:
    def __init__(self, model="env_PID", use_PID_SS=True, use_PID_CS=True):
        self.model = model
        tmp_dir = tempfile.TemporaryDirectory(prefix="model")
        tmp_path = pathlib.Path(tmp_dir.name) # временный путь до папки с временными библиотеками
        folder = pathlib.Path(__file__).parent.resolve() # папка данного скрипта
        self.dll_name = str(uuid.uuid4()) # временное название новой библиотеки
        platf = platform.system() # тип исполняющей системы (Linux или Windows)
        dll_type = '.so' if platf == 'Linux' else '_win64.dll' # тип библиотеки
        self.dll_path = os.path.join(folder, f"{model}{dll_type}") # путь к исходному файлу
        copyfile(self.dll_path, os.path.join(tmp_path, f"{self.dll_name}{dll_type}")) # копируем исходную библиотеку во временный файл
        self.dll_path = os.path.join(tmp_path, f"{self.dll_name}{dll_type}") # новый (временный) путь к библиотеке
        if platf == "Linux":
            self.dll = ctypes.cdll.LoadLibrary(self.dll_path)
        elif platform.system() == "Windows":
            self.dll = ctypes.windll.LoadLibrary(self.dll_path)
        else:
            raise Exception("Система не поддерживается")

        self.dt = 0.01 # шаг симуляции (неизменный)

        # Функции модели
        self.__initialize = getattr(self.dll, f"{model}_initialize") # функция инициализации модели
        self.__step = getattr(self.dll, f"{model}_step") # функция одного шага интегрирования
        self.__model_terminate = getattr(self.dll, f"{model}_terminate") # функция остановки модели

        # Сигналы модели
        self._state = (real_T*16).in_dll(self.dll, "state") # вектор состояния среды
        self._time = real_T.in_dll(self.dll, "sim_time") # время моделирования
        self._vartheta_ref = real_T.in_dll(self.dll, "vartheta_zh") # выходной сигнал ПИД СУ
        self._deltaz_ref = real_T.in_dll(self.dll, "deltaz_ref") # выходной сигнал ПИД СС
        self._CXa = real_T.in_dll(self.dll, "CXa")
        self._CYa  = real_T.in_dll(self.dll, "CYa")
        self._mz = real_T.in_dll(self.dll, "mz")
        self._Kalpha = real_T.in_dll(self.dll, "K_alpha")
        self._dCm_ddeltaz = real_T.in_dll(self.dll, "dCm_ddeltaz")

        # Параметры модели
        self._hzh = real_T.in_dll(self.dll, "h_zh") # требуемая высота полета
        self._use_PID_SS = boolean_T.in_dll(self.dll, "use_PID_SS") # использовать ПИД-регулятор для СС
        self._use_PID_CS = boolean_T.in_dll(self.dll, "use_PID_CS") # использовать ПИД-регулятор для СУ
        self._PID_SS = (real_T*4).in_dll(self.dll, "PID_SS")
        self._PID_CS = (real_T*4).in_dll(self.dll, "PID_CS")
        self._PID_initial = np.array(list(self._PID_CS)+list(self._PID_SS))
        self._deltaz = real_T.in_dll(self.dll, "deltaz") # угол отклонения рулей (ручное управление)
        self._vartheta_zh = real_T.in_dll(self.dll, "vartheta") # желаемый тангаж (ручное управление)

        self.use_PID_CS = use_PID_CS
        self.use_PID_SS = use_PID_SS

        self.initialize()

        self.labels = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'ax', 'ay', 'az', 'gamma', 'psi', 'vartheta', 'alpha', 'wx', 'wy', 'wz']

    #def __del__(self):
    #    del self.dll # на всякий
    #    os.remove(self.dll_path)

    def initialize(self):
        """Initialize the Model."""
        self.__initialize()
        self.step_num = -1
        self.deltaz = 0
        self.vartheta_zh = 0

    def step(self):
        """Step through the model Model."""
        self.__step()
        self.step_num += 1

    def terminate(self):
        """Terminate the model Model."""
        self.__model_terminate()

    # Signals
    @property
    def state(self) -> np.ndarray:
        return np.array(self._state)

    @property
    def state_dict(self) -> dict:
        st = self.state
        return dict((self.labels[i], st[i]) for i in range(len(self.labels)))
    
    @property
    def time(self) -> float:
        return float(self._time.value)

    @property
    def vartheta_ref(self) -> float:
        return float(self._vartheta_ref.value)

    @property
    def deltaz_ref(self) -> float:
        return float(self._deltaz_ref.value)

    @property
    def deltaz(self) -> float:
        return float(self._deltaz.value)

    @deltaz.setter
    def deltaz(self, value):
        self._deltaz.value = float(value)

    @property
    def vartheta_zh(self) -> float:
        return float(self._vartheta_zh.value)

    @vartheta_zh.setter
    def vartheta_zh(self, value):
        self._vartheta_zh.value = float(value)

    @property
    def use_PID_SS(self) -> bool:
        return bool(self._use_PID_SS.value)

    @use_PID_SS.setter
    def use_PID_SS(self, value:bool):
        self._use_PID_SS.value = value

    @property
    def use_PID_CS(self) -> bool:
        return bool(self._use_PID_CS.value)

    @use_PID_CS.setter
    def use_PID_CS(self, value:bool):
        self._use_PID_CS.value = value

    @property
    def PID_SS(self) -> np.ndarray:
        return np.array(self._PID_SS)

    @PID_SS.setter
    def PID_SS(self, value):
        for i in range(len(value)):
            self._PID_SS[i] = value[i]

    @property
    def PID_CS(self) -> np.ndarray:
        return np.array(self._PID_CS)

    @PID_CS.setter
    def PID_CS(self, value):
        for i in range(len(value)):
            self._PID_CS[i] = value[i]

    @property
    def hzh(self) -> float:
        return float(self._hzh.value)

    @hzh.setter
    def hzh(self, value):
        self._hzh.value = float(value)

    @property
    def CXa(self) -> float:
        return float(self._CXa.value)

    @property
    def CYa(self) -> float:
        return float(self._CYa.value)

    @property
    def mz(self) -> float:
        return float(self._mz.value)

    @property
    def Kalpha(self) -> float:
        return float(self._Kalpha.value)

    @property
    def dCm_ddeltaz(self) -> float:
        return float(self._dCm_ddeltaz.value)

    '''
    def __deepcopy__(self, memo):
        my_copy = type(self)(self.model, self.use_PID_SS, self.use_PID_CS)
        memo[id(self)] = my_copy
        return my_copy # поскольку ctypes запрещает делать deepcopy указателей
    '''

def main():
	pass

if __name__ == '__main__':
	main()