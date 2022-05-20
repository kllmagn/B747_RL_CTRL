import os
from typing import Union
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.chart.axis import ChartLines
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)
import pandas as pd

stepinfo_template = {
    'overshoot': None,
    'rise_time': None,
    'settling_time': None
}

model_separator = '__'

def calc_err(x1, x2) -> float:
    err = x1 - x2
    if x2 != 0:
        err /= x2
    elif x1 != 0:
        err /= x1
    else:
        err = 0
    return abs(err)

def calc_stepinfo(ys:list, y_base:float, error_band=0.05, ts:list=None):
    overshoot = (max(ys)-y_base)/y_base*100 if y_base != 0 else None #max(ys)
    try:
        tr = ts[next(i for i in range(0,len(ys)-1) if (ys[i]-ys[0])/(y_base-ys[0])>=(1-error_band))]-ts[0] if ts else None
    except StopIteration:
        tr = None
    try:
        tp = ts[next(len(ys)-i for i in range(1,len(ys)+1) if ((ys[len(ys)-i]-ys[0])/(y_base-ys[0])<=1-error_band or (ys[len(ys)-i]-ys[0])/(y_base-ys[0])>=1+error_band))]-ts[0] if ts else None
    except StopIteration:
        tp = None
    info = dict(stepinfo_template)
    info['overshoot'] = overshoot
    info['settling_time'] = tp
    info['rise_time'] = tr
    info['static_error'] = abs(ys[-1]-y_base) #)/y_base*100 if y_base != 0 else abs(ys[-1]-y_base)
    return info


class Integrator:
    def __init__(self, y0:float=0):
        self.y0 = y0
        self.sum = self.y0

    def reset(self):
        self.sum = self.y0

    def input(self, y:float):
        self.sum += y

    def output(self):
        return self.sum


class Derivative:
    def __init__(self, y0:float, dt_static=None):
        self.y0 = None
        self.y1 = y0
        self.var_step = dt_static is None
        self.dt = dt_static

    def input(self, y:float, dt=None):
        if self.var_step:
            if dt is None:
                raise ValueError("Нет информации о переменном шаге.")
            self.dt = dt
        self.y1, self.y0 = y, self.y1
        
    def output(self):
        if self.y0 is None:
            return self.y1/self.dt
        else:
            return (self.y1-self.y0)/self.dt


class DerivativeDict:
    def __init__(self):
        self.derivatives = {}

    def output(self, key:str):
        if key in self.derivatives:
            return self.derivatives[key].output()
        else:
            return 0
            #raise ValueError("Значения производной данного ключа отсутствует.")

    def input(self, key:str, y:float, dt:float):
        if key in self.derivatives:
            return self.derivatives[key].input(y)
        else:
            self.derivatives[key] = Derivative(y, dt_static=dt)
            return self.derivatives[key].output()


class MemoryDict:
    def __init__(self):
        self.memory = {}

    def input(self, key:str, y:float):
        self.memory[key] = y

    def output(self, key:str):
        return self.memory[key]

label_units = {
    'h': 'м',
    'U': 'В',
    'vartheta': 'град',
    'alpha': 'град',
    'wz': '1/с',
    'rew': '-',
    'deltaz': 'град',
    'x': 'м',
    'y': 'м',
    'V': 'м/с',
    'ax': 'м/с^2',
    'ay': 'м/с^2',
    't': 'с',
}

def comp_begin(label:str, target:str) -> bool:
    return len(label) >= len(target) and label[:len(target)] == target

def get_label_unit(label:str) -> Union[str, None]:
    for target, unit in label_units.items():
        if comp_begin(label, target):
            return f"[{unit}]"
    return None

def get_model_name_desc(model_name:str) -> str:
    description = ""
    method_to_name_mapping = {
        'obs': {
            "SPEED_MODE": "ПИД-СКОР",
            "PID_SPEED_AERO": "ПИД-СКОР-АД",
            'PID_LIKE': "ПИД",
        },
        'ctrl_mode':
        {
            "ADD_DIRECT_CONTROL": "ПКУ",
            "ADD_PROC_CONTROL": "ОКУ",
            "DIRECT_CONTROL": "ПУ",
        },
        'reset_ref_modes': {
            "CONST": "ПУТ",
            "OSCILLATING": "ОУТ",
            "HYBRID": "ГМ",
        },
        'disturbance':
        {
            'AERO_DISTURBANCE': "АД-ПОГР"
        }
    }
    for mapping in method_to_name_mapping.values():
        for name, desc in mapping.items():
            if name in model_name:
                description += ' + ' + desc if description else desc
                model_name = model_name.replace(name, "")
                break
    return description

def get_label_desc(label:str) -> str:
    if comp_begin(label, 'vartheta_ref'):
        return "Требуемый угол тангажа"
    elif comp_begin(label, "hzh"):
        return "Требуемая высота полета"
    elif model_separator not in label:
        return "СС ПИД"
    return get_model_name_desc(label)


class Storage:
    def __init__(self):
        self.storage = {}
    
    def record(self, name, value):
        if name not in self.storage:
            self.storage[name] = []
        self.storage[name].append(value)
    
    def clear(self, name):
        del self.storage[name]

    def clear_all(self, ):
        self.storage = {}

    def plot(self, names, base:str=None, xlabel=None, ylabel=None):
        if type(names) is str:
            names = [names]
        for name in names:
            if base and base in self.storage:
                plt.plot(self.storage[base], self.storage[name], label=name)
            else:
                plt.plot(self.storage[name], label=name)
        plt.grid()
        plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.show()

    def save(self, filename='storage.xls', base=None):
        print(f"Сохраняю хранилище в {filename}")
        if len(self.storage) == 0:
            raise ValueError("Невозможно сохранить хранилище: пустое хранилище")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = pd.DataFrame.from_dict(self.storage, orient='columns')

        def place_unit(label:str) -> str:
            if model_separator in label:
                parts = label.split(model_separator)
                unit = get_label_unit(parts[0])
                if unit:
                    parts[0] = f"{parts[0]}, {unit}"
                return model_separator.join(parts)
            else:
                unit = get_label_unit(label)
                if unit:
                    return f"{label}, {unit}"
                else:
                    return label

        data.columns = list(map(place_unit, data.columns))
        if base and base in self.storage:
            data.set_index(place_unit(base), inplace=True)
        writer = pd.ExcelWriter(filename)
        data.to_excel(writer, index=True, header=True, sheet_name="data")
        ws = writer.sheets['data']

        def write_chart(labels:list, pos:str):
            chart = ScatterChart()
            #chart.title = labels[0]
            #chart.style = 13
            chart.x_axis.title = 'Время t, [с]'
            chart.x_axis.minorGridlines = ChartLines()
            chart.y_axis.minorGridlines = ChartLines()
            chart.y_axis.title = labels[0]
            chart.legend.position = 'b'
            time = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            for label in labels:
                values = Reference(ws, min_col=data.columns.get_loc(label)+2, min_row=2, max_row=ws.max_row)
                series = Series(values, time, title_from_data=False, title=get_label_desc(label))
                series.smooth = True
                chart.series.append(series)
            ws.add_chart(chart, pos)

        base_labels = [label for label in data.keys() if model_separator not in label]
        for base_label in base_labels:
            child_labels = [label for label in data.keys() if (len(base_label) < len(label)) and label[:len(base_label)] == base_label and model_separator in label]
            if base_label == 'vartheta, [град]':
                child_labels.append('vartheta_ref, [град]')
            elif base_label in ['h, [м]', 'y, [м]']:
                child_labels.append('hzh, [м]')
            labels = [base_label, *child_labels]
            write_chart(labels, 'A5')

        writer.save()

    def merge(self, obj, prefix:str):
        self.storage.update(dict([(k+model_separator+prefix, v) for k,v in obj.storage.items()]))