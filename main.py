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

output_dir = '.output'

def save_best_rew_callback(agent:ControllerAgent):
    return SaveOnBestTrainingRewardCallback(
        filename=agent.bm_name,
        check_freq=log_interval,
        log_dir=agent.log_dir,
        verbose=1
    )    

def _build_env_funcs(obs_types, rew_types, ctrl_types, ctrl_modes, reset_ref_modes, disturbance_modes, tk_train, tk_test):
    envs_train, envs_test = {}, {}
    print("="*50)
    print("Построение окружений для сред.")
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
                            model_name = f"{obs_type.name if obs_type else None}_{ctrl_type.name if ctrl_type else None}_{ctrl_mode.name if ctrl_mode else None}_{reset_ref_mode.name if reset_ref_mode else None}_{disturbance_mode.name if disturbance_mode else None}"
                            print(model_name)
                            envs_train[model_name] = gen_train(obs_type, rew_type, ctrl_type, ctrl_mode, reset_ref_mode, disturbance_mode)
                            envs_test[model_name] = gen_test(obs_type, rew_type, ctrl_type, ctrl_mode, disturbance_mode)
    print("Готово!")
    print("="*50)
    return envs_train, envs_test


if __name__ == '__main__':
    obs_types = [ObservationType.PID_LIKE, ObservationType.SPEED_MODE]
    rew_types = [RewardType.CLASSIC]
    ctrl_types = [CtrlType.MANUAL]
    ctrl_modes = [CtrlMode.DIRECT_CONTROL] #, CtrlMode.ADD_DIRECT_CONTROL, CtrlMode.ADD_PROC_CONTROL]
    reset_ref_modes = [ResetRefMode.CONST] #, ResetRefMode.OSCILLATING, ResetRefMode.HYBRID]
    disturbance_modes = [None] #, DisturbanceMode.AERO_DISTURBANCE]
    tk_train = 10 # non-logging
    tk_test = 60

    train = False
    ref_values = [10*pi/180, -5*pi/180, 5*pi/180, -10*pi/180]
    use_tb = True                

    envs_train, envs_test = _build_env_funcs(obs_types, rew_types, ctrl_types, ctrl_modes, reset_ref_modes, disturbance_modes, tk_train, tk_test)

    if train:
        for model_name, env_train in envs_train.items():
            def tf_custom_recorder(agent:ControllerAgent):
                return CustomTransferProcessRecorder(
                    net_class=agent.net_class,
                    env_gen=envs_test[model_name],
                    vartheta_ref=5*pi/180,
                    state0=np.array([0, 11000, 250, 0, 0, 0]),
                    log_interval=log_interval,
                    filename=agent.bm_name,
                    log_dir=agent.log_dir,
                    window_length=100,
                    verbose=1,
                )
            callbs = [tf_custom_recorder] #, save_best_rew_callback]
            agent = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
            agent.train(env_train, 300000, callbacks_init=callbs, log_interval=int(log_interval/5))
    else:
        model_name = "COLLECTIVE_TEST" if len(envs_test) > 0 else list(envs_test.keys())[0]
        agent = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
    agent.test(ref_values, envs_test, np.array([0, 11000, 250, 0, 0, 0]), plot=False, output_dir=output_dir, collect=False)