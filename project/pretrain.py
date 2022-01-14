import tqdm

import gym

import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

from stable_baselines3 import A2C, PPO, SAC, TD3, DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

import pathlib
import pickle
import tempfile
import copy

#import seals  # noqa: F401
#import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

def create_expert_data(env_student, env_expert, verbose=1, num_interactions=4e4):
    num_interactions = int(num_interactions)
    if isinstance(env_student.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env_student.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env_student.action_space.shape[0],))
    else:
        expert_observations = np.empty((num_interactions,) + env_student.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env_student.action_space.shape)
    obs = env_expert.reset()
    for i in tqdm.tqdm(range(num_interactions)):
        action = [env_expert.envs[0].ctrl.model.deltaz_ref]
        expert_observations[i] = obs
        expert_actions[i] = action
        obs, reward, done, info = env_expert.step([1])
        if done:
            obs = env_expert.reset()
    np.savez_compressed(
        "expert_data",
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )
    expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    train_size = int(0.9 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    if verbose > 0:
        print("test_expert_dataset: ", len(test_expert_dataset))
        print("train_expert_dataset: ", len(train_expert_dataset))
    return train_expert_dataset, test_expert_dataset


def pretrain_agent(
    student,
    env_expert,
    batch_size=128,
    epochs=10,
    scheduler_gamma=0.6,
    learning_rate=1.3,
    log_interval=100,
    no_cuda=False,
    seed=1,
    test_batch_size=64,
    return_policy=False,
    verbose=1,
    **create_expert_kwargs
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    env = student.env
    train_expert_dataset, test_expert_dataset = create_expert_data(env_student=env, env_expert=env_expert, verbose=verbose, **create_expert_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                latent_pi, _, _ = model._get_latent(data)
                logits = model.action_net(latent_pi)
                action_prediction = logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if verbose > 0 and batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    latent_pi, _, _ = model._get_latent(data)
                    logits = model.action_net(latent_pi)
                    action_prediction = logits
                    target = target.long()

                test_loss = criterion(action_prediction, target)
        #test_loss /= len(test_loader.dataset)
        if verbose > 0:
            print(f"Test set: Average loss: {test_loss.item():.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    if return_policy:
        return model
    else:
        student.policy = model
        mean_reward, std_reward = evaluate_policy(student, env, n_eval_episodes=10)
        print(f"Mean reward = {mean_reward} +/- {std_reward}")
        return student


def pretrain_agent_imit(
    student,
    env_expert,
    timesteps:int=2048,
    num_episodes:int=500,
    algo='BC'
    ):
    #tempdir = tempfile.TemporaryDirectory(prefix="pretrain")
    tempdir_path = './logs/tb_log/' #pathlib.Path(tempdir.name)
    trajectories = rollout.generate_trajectories(lambda obs: [[env_expert.get_attr('ctrl')[i].model.deltaz_ref] for i in range(len(obs))], env_expert, rollout.make_sample_until(None, num_episodes))
    transitions = rollout.flatten_trajectories(trajectories)
    env = student.env
    if algo == 'BC':
        bc_logger = logger.configure(tempdir_path+"BC/")
        bc_trainer = bc.BC(
            policy=student.policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            custom_logger=bc_logger,
        )
        bc_trainer.train(n_epochs=timesteps)
        student.policy = bc_trainer.policy
    elif algo == 'AIRL':
        airl_logger = logger.configure(tempdir_path+"AIRL/")
        airl_trainer = airl.AIRL(
            venv=student.env,
            demonstrations=transitions,
            demo_batch_size=256,
            gen_algo=student,
            custom_logger=airl_logger,
        )
        airl_trainer.train(total_timesteps=timesteps)
        student.policy = airl_trainer.policy
    elif algo == 'GAIL':
        gail_logger = logger.configure(tempdir_path+"/GAIL/")
        gail_trainer = gail.GAIL(
            venv=student.env,
            demonstrations=transitions,
            demo_batch_size=256,
            gen_algo=student,
            custom_logger=gail_logger,
            #normalize_obs=False,
            #normalize_reward=False
        )
        gail_trainer.train(total_timesteps=timesteps)
        student.policy = gail_trainer.policy
    else:
        raise ValueError("Неподдерживаемый алгоритм имитационного обучения: "+algo)
    student.set_env(env)
    mean_reward, std_reward = evaluate_policy(student, student.env, n_eval_episodes=10)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")
    return student