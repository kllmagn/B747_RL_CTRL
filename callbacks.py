from math import exp, pi
import os

import numpy as np
from sqlalchemy import false
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy

from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
      

class CustomTransferProcessRecorder(BaseCallback):
  def __init__(self, env_gen, vartheta_ref:float, state0:list, log_interval:int, filename:str, log_dir:str, window_length:int=30, verbose=0):
      super(CustomTransferProcessRecorder, self).__init__(verbose)
      self.n_episodes = self.n_episodes_b = 0
      self.vartheta_ref = vartheta_ref
      self.state0 = state0
      self.log_interval = log_interval
      self.best_mean_quality = self.mean_quality = 0
      self.infos = {'settling_time': [], 'overshoot': []}
      self.log_dir = log_dir
      self.save_path = os.path.join(log_dir, filename) if filename else None
      self.env = env_gen
      self.window_length = window_length

  def calc_stepinfo(self):
    if (self.n_episodes == self.n_episodes_b):
      return
    #if self.verbose > 0:
    #    print('-'*30)
    #    print("Вычисляю ПП для vartheta_ref = ", self.vartheta_ref*180/pi)
    env = self.env()
    ctrl = env.get_attr('ctrl')[0]
    state_back, ctrl.reset_state0 = ctrl.reset_state0, self.state0
    sim_init_backup, ctrl.sim_init_state = ctrl.sim_init_state, False
    use_storage, ctrl.use_storage = ctrl.use_storage, True
    ctrl.reset_vartheta_ref = self.vartheta_ref
    obs = env.reset()
    done, state = False, None
    while not done:
      action, state = self.model.predict(obs, state=state, deterministic=True)
      obs, _, done, _ = env.step(action)
    info = ctrl.stepinfo_SS(use_backup=True)
    time, overshoot = info['settling_time'], info['overshoot']
    ctrl.reset_vartheta_ref = None
    ctrl.reset_state0 = state_back
    ctrl.sim_init_state = sim_init_backup
    ctrl.use_storage = use_storage
    _ = env.reset()
    self.infos['settling_time'].append(time)
    self.infos['overshoot'].append(overshoot)
    self.infos['settling_time'], self.infos['overshoot'] = self.infos['settling_time'][-self.window_length:], self.infos['overshoot'][-self.window_length:]
    time, overshoot = np.mean(self.infos['settling_time']), np.mean(self.infos['overshoot'])
    if time is None or overshoot is None:
      time_err = overshoot_err = np.inf
    else:
      time_err = time/ctrl.tk
      overshoot_err = 1/2*abs(overshoot)/40
    self.mean_quality = quality = exp(-time_err-overshoot_err)
    self.logger.record('transfer_custom/settling_time', time)
    self.logger.record('transfer_custom/overshoot', abs(overshoot))
    self.logger.record('transfer_custom/quality', quality)
    #if self.verbose > 0:
    #  print("Окончание вычисления")
    #  print('-'*30)

  def _on_step(self) -> bool:
      assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
      self.n_episodes_b = self.n_episodes
      self.n_episodes += np.sum(self.locals["dones"]).item()
      self.calc_stepinfo()
      if self.n_calls % self.log_interval == 0:
        if self.verbose > 0:
          print(f"Num timesteps: {self.num_timesteps}")
          print(f"Best mean transfer quality: {self.best_mean_quality:.4f} - Last mean quality per episode: {self.mean_quality:.4f}")

        # New best model, you could save the agent here
        if self.mean_quality > self.best_mean_quality:
            self.best_mean_quality = self.mean_quality
            if self.save_path:
              # Example for saving best model
              if self.verbose > 0:
                print(f"Saving new best model to {self.save_path}")
              self.model.save(self.save_path)
      return True

        
class TransferProcessRecorder(BaseCallback):
    def __init__(self, verbose=0):
        super(TransferProcessRecorder, self).__init__(verbose)
        self.n_episodes = self.n_episodes_b = 0

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes_b = self.n_episodes
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if (self.n_episodes != self.n_episodes_b):
          ctrl = self.training_env.get_attr('ctrl')[0]
          info = ctrl.stepinfo_SS(use_backup=True)
          time, overshoot = info['settling_time'], info['overshoot']
          if time is None or overshoot is None:
            tf_err = np.inf
          else:
            tf_err = time*abs(overshoot)
          self.logger.record('transfer/settling_time', time)
          self.logger.record('transfer/overshoot', abs(overshoot))
          self.logger.record('transfer/quality', 1/(1+tf_err))
        return True


class CustomEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomEvalCallback, self).__init__(verbose)
        self.n_episodes = 0
        self.n_episodes_b = 0

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes_b = self.n_episodes
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if (self.n_episodes != self.n_episodes_b):
          mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
          print(f"Mean reward = {mean_reward} +/- {std_reward}")
        return True


class EarlyStopping(BaseCallback):
    def __init__(self, metric_func, metric_name:str, check_freq: int, n_times: int, verbose: int = 1, startup_step:int = 0, maximize=True):
        super(EarlyStopping, self).__init__(verbose)
        self.check_freq = check_freq
        self.best_metric = -np.inf if maximize else np.inf
        self.n_times = n_times
        self.n_cur = 0
        self.st_step = startup_step
        self.metric_func = metric_func
        self.metric_name = metric_name
        self.maximize = maximize

    def _on_step(self) -> bool:
        if self.num_timesteps > self.st_step and self.n_calls % self.check_freq == 0:
          metric = self.metric_func()
          if ((self.maximize and self.best_metric > metric) or (not self.maximize and self.best_metric < metric)):
            self.n_cur += 1
            if self.n_cur >= self.n_times:
              return False
          else:
            self.n_cur = 0
            self.best_metric = metric
          if self.verbose > 0:
            print(f"Metric {'decreases' if self.maximize else 'increases'} [{self.metric_name}]: [{self.n_cur}/{self.n_times}]")
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, filename:str, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.mean_reward = self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              self.mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {self.mean_reward:.2f}")

              # New best model, you could save the agent here
              if self.mean_reward > self.best_mean_reward:
                  self.best_mean_reward = self.mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
        return True


class SaveOnBestQualityMetricCallback(BaseCallback):
    def __init__(self, filename:str, metric_func, metric_name:str, check_freq:int, log_dir: str, verbose: int = 1, maximize=True, mean_num:int=100):
        super(SaveOnBestQualityMetricCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.check_freq = check_freq
        self.metric_func = metric_func
        self.metric_name = metric_name
        self.maximize = maximize
        self.mean_metric = self.best_metric = -np.inf if self.maximize else np.inf
        self.mean_num = mean_num
        self.n_episodes = 0
        self.n_episodes_b = 0
        self.metric = None
        self.metrics = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes_b = self.n_episodes
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if (self.n_episodes != self.n_episodes_b): # если окончание эпизода, добавляем новое значение метрики
          self.metric = self.metric_func(self.training_env)
          self.metrics.append(self.metric)
        if (self.check_freq == -1 and self.n_episodes != self.n_episodes_b) or (self.check_freq != -1 and self.n_calls % self.check_freq == 0):
          self.mean_metric = np.mean(self.metrics[-self.mean_num:])
          if self.verbose > 0:
            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best metric [{self.metric_name}]: {self.best_metric:.2f} - Last mean metric per episode: {self.mean_metric:.2f}")
          if (self.maximize and self.mean_metric > self.best_metric) or (not self.maximize and self.mean_metric < self.best_metric):
            self.best_metric = self.mean_metric
            if self.verbose > 0:
              print(f"Saving new best model to {self.save_path}")
            self.model.save(self.save_path)
        return True


class ExecOnEpisodeEnd(BaseCallback):
    def __init__(self, exec_func, verbose: int = 1):
        super(SaveOnBestQualityMetricCallback, self).__init__(verbose)
        self.n_episodes = 0
        self.n_episodes_b = 0
        self.exec_func = exec_func

    def _on_step(self) -> bool:
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes_b = self.n_episodes
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if (self.n_episodes != self.n_episodes_b): # если окончание эпизода
          if self.verbose:
            print(f'Executing function on episode {self.n_episodes}')
          self.exec_func(self.training_env)
        return True