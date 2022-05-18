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

def train_test(obs_type:ObservationType, rew_type:RewardType, ctrl_type:CtrlType, ctrl_mode:CtrlMode, tk:float, reset_ref_mode:ResetRefMode, disturbance_mode:DisturbanceMode):
    def env_train():
        return ControllerEnv(
            obs_type,
            rew_type,
            norm_obs,
            norm_act,
            ctrl_type,
            ctrl_mode,
            tk = tk,
            reset_ref_mode = reset_ref_mode,
            disturbance_mode = disturbance_mode,
            sample_time = sample_time,
            use_limiter = False,
            action_max=ctrl_mode_max[ctrl_mode],
            logging_path=controller_log_path
        )

    def env_test():
        return ControllerEnv(
            obs_type,
            rew_type,
            norm_obs,
            norm_act,
            ctrl_type,
            ctrl_mode,
            tk = 60,
            reset_ref_mode = None,
            disturbance_mode = disturbance_mode,
            sample_time = sample_time,
            use_limiter = False,
            action_max=ctrl_mode_max[ctrl_mode],
            aero_err=aero_err_test,
        )

    def tf_custom_recorder(agent:ControllerAgent):
        return CustomTransferProcessRecorder(
            net_class=agent.net_class,
            env_gen=env_test,
            vartheta_ref=5*pi/180,
            state0=np.array([0, 11000, 250, 0, 0, 0]),
            log_interval=log_interval,
            filename=agent.bm_name,
            log_dir=agent.log_dir,
            window_length=100,
            verbose=1,
        )
    callbs = [tf_custom_recorder] #, save_best_rew_callback]
    use_tb = True
    model_name = f"{obs_type.name if obs_type else None}_{ctrl_type.name if ctrl_type else None}_{ctrl_mode.name if ctrl_mode else None}_{reset_ref_mode.name if reset_ref_mode else None}_{disturbance_mode.name if disturbance_mode else None}"
    agent = ControllerAgent(net_class=net_class, use_tb=use_tb, model_name=model_name)
    print(f"Тренировка модели подхода: {model_name}")
    agent.train(env_train, timesteps=300000, callbacks_init=callbs, log_interval=int(log_interval/5))
    print(f"Тестирование модели подхода: {model_name}")
    agent.test([10*pi/180, -5*pi/180, 5*pi/180, -10*pi/180], env_test, state0=np.array([0, 11000, 250, 0, 0, 0]), plot=True, output_dir='.output')


if __name__ == '__main__':
    obs_types = [ObservationType.PID_LIKE] #, ObservationType.SPEED_MODE, ObservationType.PID_SPEED_AERO]
    ctrl_modes = [CtrlMode.ADD_DIRECT_CONTROL] #CtrlMode.DIRECT_CONTROL, CtrlMode.ADD_DIRECT_CONTROL, CtrlMode.ADD_PROC_CONTROL]
    reset_ref_modes = [ResetRefMode.CONST] #ResetRefMode.CONST, ResetRefMode.OSCILLATING, ResetRefMode.HYBRID]
    disturbance = None #DisturbanceMode.AERO
    tk_train = 10 # non-logging

    for obs_type in obs_types:
        for ctrl_mode in ctrl_modes:
            for reset_ref_mode in reset_ref_modes:
                #try:
                train_test(obs_type, RewardType.CLASSIC, CtrlType.MANUAL, ctrl_mode, tk_train, reset_ref_mode, disturbance)
                #except Exception as e:
                #    print(f"Ошибка при обучении/тестировании модели [{obs_type, ctrl_mode, reset_ref_mode}]:", e)
                #break