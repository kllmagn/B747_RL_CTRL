import ctypes
import os
import platform
import pathlib
from shutil import copyfile
import tempfile
import pathlib
import typing
import weakref

import uuid
import numpy as np
from torch import ge

from .rtwtypes import real_T, boolean_T

def generate_param(ref_name:str, param_type, getter_filter:typing.Union[typing.Callable, list]=None):
    is_array = param_type in [list, np.ndarray, np.array] # array like
    if param_type is np.ndarray:
        param_type = np.array
    @property
    def param(self:Model) -> param_type:
        if is_array:
            val = param_type(list(self.__getattribute__(ref_name)))
        else:
            val = param_type(self.__getattribute__(ref_name).value)
        if getter_filter is not None:
            if type(getter_filter) is list:
                for gf in getter_filter:
                    val = gf(val)
            else:
                val = getter_filter(val)
        return val
    @param.setter
    def param(self:Model, value):
        if is_array:
            for i in range(len(value)):
                self.__getattribute__(ref_name)[i] = value[i]
        else:
            self.__getattribute__(ref_name).value = param_type(value)
    return param

def generate_signal(ref_name:str, param_type, getter_filter:typing.Union[typing.Callable, list]=None):
    is_array = param_type in [list, np.ndarray, np.array]
    if param_type is np.ndarray:
        param_type = np.array
    @property
    def signal(self:Model) -> param_type:
        if is_array:
            val = param_type(list(self.__getattribute__(ref_name)))
        else:
            val = param_type(self.__getattribute__(ref_name).value)
        if getter_filter is not None:
            if type(getter_filter) is list:
                for gf in getter_filter:
                    val = gf(val)
            else:
                val = getter_filter(val)
        return val
    return signal

class Model:
    def __init__(self, model="model", use_PID_SS=True, use_PID_CS=True, initial_state:np.ndarray=None):
        self.model = model
        folder = pathlib.Path(__file__).parent.resolve() # папка данного скрипта
        self.tmp_dir = os.path.join(folder, 'tmp_models') #tempfile.TemporaryDirectory(prefix="model")
        tmp_path = self.tmp_dir #pathlib.Path(self.tmp_dir.name) # временный путь до папки с временными библиотеками
        os.makedirs(tmp_path, exist_ok=True)
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

        # Сигналы модели Simulink
        self._state = (real_T*16).in_dll(self.dll, "state") # вектор состояния среды
        self._time = real_T.in_dll(self.dll, "sim_time") # время моделирования
        self._vartheta_ref = real_T.in_dll(self.dll, "vartheta_zh") # выходной сигнал ПИД СУ
        self._deltaz_ref = real_T.in_dll(self.dll, "deltaz_ref") # выходной сигнал ПИД СС
        self._CXa = real_T.in_dll(self.dll, "CXa")
        self._CYa  = real_T.in_dll(self.dll, "CYa")
        self._mz = real_T.in_dll(self.dll, "mz")
        self._Kalpha = real_T.in_dll(self.dll, "K_alpha")
        self._dCm_ddeltaz = real_T.in_dll(self.dll, "dCm_ddeltaz")

        # Параметры модели Simulink
        self._state0 = (real_T*16).in_dll(self.dll, 'state0') # начальное состояние модели
        self._hzh = real_T.in_dll(self.dll, "h_zh") # требуемая высота полета
        self._use_PID_SS = real_T.in_dll(self.dll, "use_PID_SS") # использовать ПИД-регулятор для СС
        self._use_PID_CS = real_T.in_dll(self.dll, "use_PID_CS") # использовать ПИД-регулятор для СУ
        self._PID_SS = (real_T*4).in_dll(self.dll, "PID_SS")
        self._PID_CS = (real_T*4).in_dll(self.dll, "PID_CS")
        self._deltaz = real_T.in_dll(self.dll, "deltaz") # угол отклонения рулей (ручное управление)
        self._vartheta_zh = real_T.in_dll(self.dll, "vartheta") # желаемый тангаж (ручное управление)
        self._P = real_T.in_dll(self.dll, 'P') # ручное значение тяги

        # Фильтры сигналов
        def remove_nan(val):
            return np.nan_to_num(val)

        # Сигналы класса
        Model.time = generate_signal('_time', float)
        Model.vartheta_ref = generate_signal('_vartheta_ref', float)
        Model.deltaz_ref = generate_signal('_deltaz_ref', float)
        Model.CXa = generate_signal('_CXa', float)
        Model.CYa = generate_signal('_CYa', float)
        Model.mz = generate_signal('_mz', float)
        Model.Kalpha = generate_signal('_Kalpha', float)
        Model.dCm_ddeltaz = generate_signal('_dCm_ddeltaz', float)

        Model.state0 = generate_param('_state0', np.ndarray) # ПАРАМЕТР
        def set_initial(val):
            nonlocal self
            if self.time == 0:
                return self.state0
            else:
                return val
        Model.state = generate_signal('_state', np.ndarray, getter_filter=[set_initial, remove_nan])

        # Параметры класса
        Model.hzh = generate_param('_hzh', float)
        Model.use_PID_SS = generate_param('_use_PID_SS', float)
        Model.use_PID_CS = generate_param('_use_PID_CS', float)
        Model.PID_SS = generate_param('_PID_SS', np.ndarray)
        Model.PID_CS = generate_param('_PID_CS', np.ndarray)
        Model.deltaz = generate_param('_deltaz', float)
        Model.vartheta_zh = generate_param('_vartheta_zh', float)
        Model.P = generate_param('_P', float)

        self._PID_initial = np.array(list(self._PID_CS)+list(self._PID_SS))
        # назначение конфигурации в соответствии с аргументами
        if initial_state is not None:
            self.state0 = initial_state
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

    @property
    def state_dict(self) -> dict:
        st = self.state
        return dict((self.labels[i], st[i]) for i in range(len(self.labels)))
    

def main():
    model = Model(use_PID_CS=True, initial_state=np.array([100, 1000, 50, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    model.hzh = 12000
    model.P = 300000
    model.vartheta_zh = 0.1
    print(model.time, model.state_dict)
    while model.time < 2000:
        model.step()
        #print(model.time, model.state)
    print(model.time, model.state_dict)

if __name__ == '__main__':
	main()