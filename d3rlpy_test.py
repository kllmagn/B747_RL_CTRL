from tqdm import tqdm

import gym
from stable_baselines3.common.evaluation import evaluate_policy

from d3rlpy.algos import PLAS
from d3rlpy.wrappers.sb3 import SB3Wrapper

from nirs.envs.ctrl_env.ctrl_env import ControllerEnv

env = ControllerEnv(use_storage=True, is_testing=True)

offline_model = PLAS(use_gpu=False, batch_size=4)
model = SB3Wrapper(offline_model)

offline_model.fit_online(env, n_steps=2000)

mean_reward, std_reward = evaluate_policy(model, env)
print(f"mean_reward={mean_reward} +/- {std_reward}")

done = False
obs = env.reset()
data = {}
state = None
num_interactions = 4999

for i in tqdm(range(num_interactions), desc="Тестирование модели"):
    action, state = model.predict(obs, state=state, deterministic=True)
    obs, reward, done, data = env.step(action)
    if use_render:
        env.render()

env.ctrl.storage.plot(['vartheta', 'vartheta_ref'], 't')