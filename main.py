from stable_baselines3 import A2C
from core.controller import DisturbanceMode, ResetRefMode, CtrlType, CtrlMode
from neural.agent import *

from math import pi

from neural.setups import TrainPlotter

ctrl_mode_max = {
    CtrlMode.DIRECT_CONTROL: 17*pi/180,
    CtrlMode.ANG_VEL_CONTROL: 2*pi/180,
    CtrlMode.ADD_PROC_CONTROL: 0.7,
    CtrlMode.ADD_DIRECT_CONTROL: 10*pi/180,
}

net_class = A2C
norm_obs = True
norm_act = True
aero_err_test = np.array([-0.1, 0.1, -0.1, -0.1, 0.1])
sample_time = 0.05
controller_log_path = "./controller.log"
log_interval = 1000

def save_best_rew_callback(agent:ControllerAgent):
    return SaveOnBestTrainingRewardCallback(
        filename=agent.bm_name,
        check_freq=log_interval,
        log_dir=agent.log_dir,
        verbose=1
    )    

def _build_env_funcs(obs_types, rew_types, ctrl_types, ctrl_modes, reset_ref_modes, disturbance_modes, tk_train, tk_test, drop_train_disturb=False, drop_test_disturb=False):
    envs_train, envs_test = {}, {}
    print("="*50)
    print("Построение окружений для сред.")
    if drop_train_disturb and drop_test_disturb:
        raise ValueError("Нельзя сбрасывать возмущения у обоих окружений.")
    elif drop_train_disturb:
        print("Игнорирую параметр возмущения при составлении названия окружения обучения.")
    elif drop_test_disturb:
        print("Игнорирую параметр возмущения при составлении названия окружения тестирования.")
    def gen_train(obs_type, rew_type, ctrl_type, ctrl_mode, reset_ref_mode, disturbance_mode):
        return lambda: ControllerEnv(
                                obs_type,
                                rew_type,
                                norm_obs,
                                norm_act,
                                ctrl_type,
                                ctrl_mode,
                                tk = tk_train,
                                reset_ref_mode = reset_ref_mode,
                                disturbance_mode = disturbance_mode,
                                sample_time = sample_time,
                                use_limiter = False,
                                action_max=ctrl_mode_max[ctrl_mode],
                                logging_path=controller_log_path
                            )
    def gen_test(obs_type, rew_type, ctrl_type, ctrl_mode, disturbance_mode):
        return lambda: ControllerEnv(
                                obs_type,
                                rew_type,
                                norm_obs,
                                norm_act,
                                ctrl_type,
                                ctrl_mode,
                                tk = tk_test,
                                reset_ref_mode = None,
                                disturbance_mode = disturbance_mode,
                                sample_time = sample_time,
                                use_limiter = False,
                                action_max=ctrl_mode_max[ctrl_mode],
                                aero_err=aero_err_test,
                            )
    for obs_type in obs_types:
        for rew_type in rew_types:
            for ctrl_type in ctrl_types:
                for ctrl_mode in ctrl_modes:
                    for reset_ref_mode in reset_ref_modes:
                        for disturbance_mode in disturbance_modes:
                            model_name_train = f"{obs_type.name if obs_type else None}_{ctrl_type.name if ctrl_type else None}_{ctrl_mode.name if ctrl_mode else None}_{reset_ref_mode.name if reset_ref_mode else None}_{disturbance_mode.name if (disturbance_mode and not drop_train_disturb) else None}"
                            model_name_test = f"{obs_type.name if obs_type else None}_{ctrl_type.name if ctrl_type else None}_{ctrl_mode.name if ctrl_mode else None}_{reset_ref_mode.name if reset_ref_mode else None}_{disturbance_mode.name if (disturbance_mode and not drop_test_disturb) else None}"
                            print("Название окружений:", model_name_train, "|", model_name_test)
                            envs_train[model_name_train] = gen_train(obs_type, rew_type, ctrl_type, ctrl_mode, reset_ref_mode, disturbance_mode)
                            envs_test[model_name_test] = gen_test(obs_type, rew_type, ctrl_type, ctrl_mode, disturbance_mode)
    print(f"Готово! Количество окружений: {len(envs_train)}")
    print("="*50)
    return envs_train, envs_test


if __name__ == '__main__':
    obs_types = [ObservationType.PID_LIKE, ObservationType.SPEED_MODE]
    rew_types = [RewardType.CLASSIC]
    ctrl_types = [CtrlType.MANUAL]
    ctrl_modes = [CtrlMode.DIRECT_CONTROL, CtrlMode.ADD_DIRECT_CONTROL, CtrlMode.ADD_PROC_CONTROL]
    reset_ref_modes = [ResetRefMode.CONST, ResetRefMode.OSCILLATING, ResetRefMode.HYBRID]
    disturbance_modes = [None] #DisturbanceMode.AERO_DISTURBANCE]
    tk_train = 20 # non-logging
    tk_test = 20

    pid_coefs = [
        #np.array([-5.9151, -1.2404, -6.6927, 58.0826]),
        #np.array([-5.9151, -1.2804, -7.2227, 58.0826]),
        #np.array([-7.3151, -0.3404, -6.6927, 58.0826]),
        #np.array([-4.9217, -8.2404, -6.6927, 58.0826]), # раб время ПП
        #np.array([-10.7217, -2.591, -7.7875, 258.0826]), #661.7417]), # tp = 3,17 tuned, 6,885 - actual?
        #np.array([-6.4344, -1.2133, -7.731, 258.0826]), #650.4214]), # overshoot = 7,81 tuned, 7,143801 - actual?
    ]

    train = True
    train_timesteps = 300000
    optimize = False
    verbose = 1
    preload = False
    ref_values = [5*pi/180, -5*pi/180, 10*pi/180, -10*pi/180]
    use_tb = True
    drop_train_disturb = False # обучить без возмущений
    drop_test_disturb = False # протестировать модели, обученные без возмущений на средах с возмущениями
    output_dir = '.output'
    no_neural = False # не тестировать нейросетевые модели (только ПИД)
    plot = False
    reward_config = {} #{'rmax': 0.4650050136731758, 'k1': 1.6611237678124193, 'k2': 0.1486181402065907}

    sm0 = np.array([0, 11000, 250, 0, 0, 0]) # начальный вектор состояния моделирования при тестировании

    envs_train, envs_test = _build_env_funcs(
        obs_types,
        rew_types,
        ctrl_types,
        ctrl_modes,
        reset_ref_modes,
        disturbance_modes,
        tk_train,
        tk_test,
        drop_train_disturb=drop_train_disturb,
        drop_test_disturb=drop_test_disturb,
    )

    if train:
        for model_name, env_train in envs_train.items():
            def control_test(agent:ControllerAgent):
                return ControlTestCallback(
                    net_class=agent.net_class,
                    env_gen=envs_test[model_name],
                    vartheta_ref=ref_values,
                    state0=sm0,
                    log_interval=log_interval,
                    filename=agent.bm_name,
                    log_dir=agent.log_dir,
                    window_length=30,
                    verbose=1,
                )
            callbs = [control_test] #, save_best_rew_callback]
            agent = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
            agent.train(env_train, train_timesteps, callbacks_init=callbs, log_interval=int(log_interval/5), optimize=optimize, preload=preload, reward_config=reward_config, verbose=verbose)
    else:
        model_name = "COLLECTIVE_TEST" if len(envs_test) > 1 else list(envs_test.keys())[0]
        agent = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
    agent.test(ref_values, envs_test, sm0, plot=plot, output_dir=output_dir, collect=False, no_neural=no_neural, pid_coefs=pid_coefs)