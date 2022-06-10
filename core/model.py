import ctypes
from ctypes import wintypes
import logging
import os
import platform
import pathlib
from shutil import copyfile, rmtree
import tempfile
import pathlib
import typing

import uuid
import numpy as np

if __name__ == "__main__":
    from rtwtypes import real_T
else:
    from .rtwtypes import real_T

def on_exit():
    folder = pathlib.Path(__file__).parent.resolve() # папка данного скрипта
    tmp_dir = os.path.join(folder, 'tmp_models')
    rmtree(tmp_dir)

import win32api
win32api.SetConsoleCtrlHandler(on_exit, True)

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)    
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]


def generate_param(ref_name:str, param_type, getter_filter:typing.Union[typing.Callable, list]=None):
    is_array = param_type in [list, np.ndarray, np.array] # array like
    if param_type is np.ndarray:
        param_type = np.array
    @property
    def param(self:Model) -> param_type:
        self.logger.debug(f"Получение значения параметра {ref_name}.")
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
        self.logger.debug(f"Установка параметра {ref_name}: {value}")
        if is_array:
            for i in range(len(value)):
                self.__getattribute__(ref_name)[i] = value[i]
        else:
            self.__getattribute__(ref_name).value = param_type(value)
    return param


def generate_signal(ref_name:str, param_type, getter_filter:typing.Union[typing.Callable, list]=None) -> typing.Callable[['Model'], typing.Any]:
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


class ModelLogger():
    @property
    def logger(self):
        return logging.Logger(f"[Model]", logging.INFO)


class Model(ModelLogger):
    def __init__(self, model="model_simple", use_PID_SS:bool=True, use_PID_CS:bool=True, initial_state:np.ndarray=None, logging_path="model.log", use_RP:bool=True):
        model_id = str(uuid.uuid4())
        log_level = logging.INFO
        #self.logger = logging.Logger(f"[Model_{model_id}]", log_level)
        if logging_path:
            fh = logging.FileHandler(logging_path, 'a+', encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            fh.setLevel(log_level)
            self.logger.addHandler(fh)
        self.logger.info(f"Инициализация модели: model = {model}, use_PID_CS = {use_PID_CS}, use_PID_SS = {use_PID_SS}, use_RP = {use_RP}, initial_state = {initial_state}.")
        self.model = model
        folder = pathlib.Path(__file__).parent.resolve() # папка данного скрипта
        self.tmp_dir = os.path.join(folder, 'tmp_models') #tempfile.TemporaryDirectory(prefix="model")
        #pathlib.Path(self.tmp_dir.name) # временный путь до папки с временными библиотеками
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.dll_name = model_id # временное название новой библиотеки
        platf = platform.system() # тип исполняющей системы (Linux или Windows)
        dll_type = '.so' if platf == 'Linux' else '_win64.dll' # тип библиотеки
        self.dll_path = os.path.join(folder, f"{model}{dll_type}") # путь к исходному файлу
        tmp_path = os.path.join(self.tmp_dir, f"{self.dll_name}{dll_type}") # новый (временный) путь к библиотеке
        self.logger.info(f"Временное копирование библиотеки: {self.dll_path} -> {tmp_path}")
        copyfile(self.dll_path, tmp_path) # копируем исходную библиотеку во временный файл
        self.dll_path = tmp_path
        self.logger.info(f"({platf}) Поиск подключаемой библиотеки: {self.dll_path}")
        if platf == "Linux":
            self.dll = ctypes.cdll.LoadLibrary(self.dll_path)
        elif platform.system() == "Windows":
            self.dll = ctypes.windll.LoadLibrary(self.dll_path)
        else:
            self.logger.error(f"Указанная система не поддерживается: {platf}")
            raise Exception("Система не поддерживается")
        self.logger.info("Успешная загрузка библиотеки.")

        self.dt = 0.01 # шаг симуляции (неизменный)

        # Функции модели
        self.__initialize = getattr(self.dll, f"{model}_initialize") # функция инициализации модели
        self.__step = getattr(self.dll, f"{model}_step") # функция одного шага интегрирования
        self.__model_terminate = getattr(self.dll, f"{model}_terminate") # функция остановки модели

        # Сигналы модели Simulink
        self._state = (real_T*6).in_dll(self.dll, "state") # вектор состояния среды
        self._time = real_T.in_dll(self.dll, "sim_time") # время моделирования
        self._vartheta_ref = real_T.in_dll(self.dll, "vartheta_zh") # выходной сигнал ПИД СУ
        self._deltaz_ref = real_T.in_dll(self.dll, "U_com_PID") # выходной сигнал ПИД СС
        self._CXa = real_T.in_dll(self.dll, "CXa")
        self._CYa  = real_T.in_dll(self.dll, "CYa")
        self._mz = real_T.in_dll(self.dll, "mz")
        self._Kalpha = real_T.in_dll(self.dll, "K_alpha")
        self._dCm_ddeltaz = real_T.in_dll(self.dll, "dCm_ddeltaz")
        self._deltaz_com = real_T.in_dll(self.dll, 'U_com') # командный угол отклонения рулей (вход рулевого привода / без насыщения)
        self._deltaz_real = real_T.in_dll(self.dll, 'deltaz_RP') # реальный угол отклонения рулей (выход рулевого привода)
        self._dvartheta = real_T.in_dll(self.dll, 'dvartheta')
        self._dvartheta_int = real_T.in_dll(self.dll, 'dvartheta_int')
        self._dvartheta_dt = real_T.in_dll(self.dll, 'dvartheta_dt')
        self._dvartheta_dt_dt = real_T.in_dll(self.dll, 'dvartheta_dt_dt')
        self._TAE = real_T.in_dll(self.dll, 'TAE')
        self._ITAE = real_T.in_dll(self.dll, 'ITAE')
        self._TSE = real_T.in_dll(self.dll, 'TSE')
        self._ITSE = real_T.in_dll(self.dll, 'ITSE')
        self._AE = real_T.in_dll(self.dll, 'AE')
        self._IAE = real_T.in_dll(self.dll, 'IAE')
        self._SE = real_T.in_dll(self.dll, 'SE')
        self._ISE = real_T.in_dll(self.dll, 'ISE')
        
        # Параметры модели Simulink
        self._state0 = (real_T*6).in_dll(self.dll, 'state0') # начальное состояние модели
        self._hzh = real_T.in_dll(self.dll, "h_zh") # требуемая высота полета
        self._use_RP = real_T.in_dll(self.dll, "use_RP") # использовать рулевой привод
        self._use_PID_SS = real_T.in_dll(self.dll, "use_PID_SS") # использовать ПИД-регулятор для СС
        self._use_PID_CS = real_T.in_dll(self.dll, "use_PID_CS") # использовать ПИД-регулятор для СУ
        self._PID_SS = (real_T*4).in_dll(self.dll, "PID_SS")
        self._PID_CS = (real_T*4).in_dll(self.dll, "PID_CS")
        self._deltaz = real_T.in_dll(self.dll, "deltaz") # угол отклонения рулей (ручное управление)
        self._vartheta_zh = real_T.in_dll(self.dll, "vartheta") # желаемый тангаж (ручное управление)
        self._P = real_T.in_dll(self.dll, 'P') # ручное значение тяги
        self._aero_err = (real_T*5).in_dll(self.dll, 'aero_err') # вектор ошибок а/д коэффициентов в диапазоне [-aero_err_max; aero_err_max] (CXa, CYa, mz, mz_deltaz, Kalpha)

        # Фильтры сигналов
        def remove_nan(val):
            return np.nan_to_num(val)

        # Сигналы класса
        Model.time = generate_signal('_time', float)
        Model.vartheta_ref = generate_signal('_vartheta_ref', float)
        Model.deltaz_ref = generate_signal('_deltaz_ref', float)
        Model.deltaz_com = generate_signal('_deltaz_com', float)
        Model.deltaz_real = generate_signal('_deltaz_real', float)
        Model.CXa = generate_signal('_CXa', float)
        Model.CYa = generate_signal('_CYa', float)
        Model.mz = generate_signal('_mz', float)
        Model.Kalpha = generate_signal('_Kalpha', float)
        Model.dCm_ddeltaz = generate_signal('_dCm_ddeltaz', float)
        Model.dvartheta = generate_signal('_dvartheta', float)
        Model.dvartheta_int = generate_signal('_dvartheta_int', float)
        Model.dvartheta_dt = generate_signal('_dvartheta_dt', float)
        Model.dvartheta_dt_dt = generate_signal('_dvartheta_dt_dt', float)
        Model.TAE = generate_signal('_TAE', float)
        Model.ITAE = generate_signal('_ITAE', float)
        Model.TSE = generate_signal('_TSE', float)
        Model.ITSE = generate_signal('_ITSE', float)
        Model.AE = generate_signal('_AE', float)
        Model.IAE = generate_signal('_IAE', float)
        Model.SE = generate_signal('_SE', float)
        Model.ISE = generate_signal('_ISE', float)

        Model.state0 = generate_param('_state0', np.ndarray) # ПАРАМЕТР
        def set_initial(val):
            if self.time == 0:
                return self.state0
            else:
                return val
        Model.state = generate_signal('_state', np.ndarray, getter_filter=[remove_nan])

        # Параметры класса
        Model.hzh = generate_param('_hzh', float)
        Model.use_RP = generate_param('_use_RP', float)
        Model.use_PID_SS = generate_param('_use_PID_SS', float)
        Model.use_PID_CS = generate_param('_use_PID_CS', float)
        Model.PID_SS = generate_param('_PID_SS', np.ndarray)
        Model.PID_CS = generate_param('_PID_CS', np.ndarray)
        Model.deltaz = generate_param('_deltaz', float)
        Model.vartheta_zh = generate_param('_vartheta_zh', float)
        Model.P = generate_param('_P', float)
        Model.aero_err = generate_param('_aero_err', np.ndarray)

        self._PID_initial = np.array(list(self._PID_CS)+list(self._PID_SS))
        # назначение конфигурации в соответствии с аргументами
        if initial_state is not None:
            self.state0 = initial_state
        self.use_RP = use_RP
        self.use_PID_CS = use_PID_CS
        self.use_PID_SS = use_PID_SS

        self.Pmax = self.P
        
        self.initialize()

        self.labels = ['x', 'y', 'Vx', 'Vy', 'vartheta', 'wz']


    def __del__(self):
        self.logger.info("Удаление объекта модели.")
        if platform.system() == 'Windows':
            handle = self.dll._handle
            del self.dll # на всякий
            kernel32.FreeLibrary(handle)
            os.remove(self.dll_path)


    def initialize(self):
        """Initialize the Model."""
        self.logger.info("Начальная инициализация модели.")
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
        self.logger.info("Принудительная остановка модели.")
        self.__model_terminate()


    def set_initial(self, state:np.ndarray):
        self.state0 = state
        self.logger.info(f"Установка начального состояния: {self.state0}")


    @property
    def state_dict(self) -> dict:
        st = self.state
        return dict((self.labels[i], st[i]) for i in range(len(self.labels)))
    

def main():
    model = Model(use_PID_CS=False, initial_state=np.array([100, 1000, 300, 0, 0, 0]))
    model.hzh = 2000
    model.P = 300000
    model.vartheta_zh = -0.1
    print(model.time, model.state_dict)
    while model.time < 2000:
        model.step()
        print(model.time, model.state_dict['vartheta'], model.state_dict['y'], model.deltaz_real)
    print(model.time, model.state_dict)

if __name__ == '__main__':
	main()