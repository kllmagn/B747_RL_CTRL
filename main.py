from asyncore import write
from threading import Thread
from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
from ctrl_agent import *

from math import pi

from setups import TrainPlotter

ctrl_max_modes = {
    CtrlMode.DIRECT_CONTROL: 17*pi/180,
    CtrlMode.ANG_VEL_CONTROL: 2*pi/180,
    CtrlMode.ADD_PROC_CONTROL: 0.5,
    CtrlMode.ADD_DIRECT_CONTROL: 5*pi/180,
}

if __name__ == '__main__':
    net_class = A2C
    ctrl_mode = CtrlMode.DIRECT_CONTROL
    use_tb = True
    model_name = 'NOSIM_DIRECT_CONTROL_AERR'
    test_output_dir = 'output'
    log_interval = 1000
    plot_test = True

    env_kwargs = dict(
        use_ctrl = False, # использовать СУ (ПИД-регулятор авто или коррекция)
        manual_ctrl = False, # вкл. ручное управление СУ (откл. поддержку ПИД-регулятора)
        manual_stab = True, # вкл. ручное управление СС (откл. поддержку ПИД-регулятора)
        no_correct = True, # не использовать коррекцию коэффициентов ПИД-регуляторов
        sample_time = 0.05,
        use_limiter = False,
        ctrl_mode=ctrl_mode,
        action_max=ctrl_max_modes[ctrl_mode],
        #reward_config={'k1': 0.9, 'k2': 0.3, 'k3': 0.4}, #{'k1': 0.5626263389608758, 'k2': 0.957988620443826, 'k3': 0.20433884176957848} #{'kv': 23.02559907773439, 'kw': 123.40541803849644, 'kdeltaz': 6.523852550774975}
    )
    # ===== Имитационное обучение ======
    pretrain = False 
    pretrain_kwargs = dict(
        timesteps = 1_000_000, # epochs (BC)
        preload = False,
        num_int_episodes = 200,
        algo = 'GAIL' # BC, GAIL, AIRL
    )
    # ============ Обучение =============
    train = False
    train_kwargs = dict(
        timesteps = 800000,
        tk = 20, # секунд
        preload = False,
        use_es = False,
        optimize = False,
        opt_max = True,
        opt_hp = True,
        verbose = int(use_tb),
        log_interval = log_interval,
        show_plotter = False,
        #use_storage = False,
        random_init = True, # случайная инициализация начального состояния
        sim_init_state = False
    )
    # ========== Тестирование ==========
    test_kwargs = dict(
        tk = 60, # секунд
        random_init = True, # случайная инициализация начального состояния
        sim_init_state = False,
        reset_state0 = [0, 11000, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        reset_aero_err = [0.2, 0.1, -0.2, -0.2]
    )
    # ==================================
    ctrl = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
    if pretrain:
        ctrl.pretrain(**pretrain_kwargs, **env_kwargs)
    if train:
        ctrl.train(**train_kwargs, **env_kwargs)
    #ctrl.test(**test_kwargs, **env_kwargs)
    # ==================================
    varthetas = [5*pi/180] #, 10*pi/180, -10*pi/180, -5*pi/180]
    hs = [] #10000, 10500, 11500, 12000]
    def write_str(path:str, content:str):
        f = open(path, 'w')
        f.write(content)
        f.close()
    for i in range(len(varthetas)):
        print('='*30)
        print('Тестирую угол тангажа vartheta =', varthetas[i]*180/pi, '[град]')
        env_kwargs['use_ctrl'] = False
        storage, info_neural, info_pid = ctrl.test(ht_func = lambda t: 11000, varthetat_func = lambda t: varthetas[i], plot=plot_test, **test_kwargs, **env_kwargs)
        storage.save(f'{test_output_dir}/{model_name}/data_vartheta_{varthetas[i]*180/pi}.xlsx')
        write_str(f'{test_output_dir}/{model_name}/data_vartheta_{varthetas[i]*180/pi}_info.txt', 'НС:'+str(info_neural)+'\n'+'ПИД:'+str(info_pid))
    for i in range(len(hs)):
        print('='*30)
        print('Тестирую высоту h =', hs[i], '[м]')
        env_kwargs['use_ctrl'] = True
        storage, info_neural, info_pid = ctrl.test(ht_func = lambda t: hs[i], varthetat_func = lambda t: 0, plot=plot_test, **test_kwargs, **env_kwargs)
        storage.save(f'{test_output_dir}/{model_name}/data_h_{hs[i]}.xlsx')
        write_str(f'{test_output_dir}/{model_name}/data_h_{hs[i]}_info.txt', 'НС:'+str(info_neural)+'\n'+'ПИД:'+str(info_pid))
    ctrl.show()
    ctrl.convert_to_onnx('model.onnx')
    ctrl.test_onnx('model.onnx')