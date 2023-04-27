import argparse
import copy
import json
import os

from datetime import datetime
from pathlib import Path

import json
import pyaml
import torch
import yaml
import numpy as np

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import pybullet_data
import pybullet_envs  # register pybullet envs from bullet3

import NerveNet.gym_envs.pybullet.register_disability_envs

import gym
from gym import wrappers
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from util import LoggingCallback
algorithms = dict(A2C=A2C, PPO=PPO)


def create_video(args):
    with open(args.train_output / "train_arguments.yaml") as yaml_data:
        train_arguments = yaml.load(yaml_data,
                                    Loader=yaml.FullLoader)

    model = algorithms[train_arguments["alg"]].load(
        args.train_output / "".join(train_arguments["model_name"].split(".")[:-1]), device='cuda:0')
    # base_xml_path_parts = model.policy.mlp_extractor.xml_assets_path.parents._parts
    # if "pybullet_data" in base_xml_path_parts:
    #     model.policy.mlp_extractor.xml_assets_path.parents._parts = Path(
    #         pybullet_data.getDataPath()) / "mjcf"

    env_name = train_arguments["task_name"]
    env = gym.make(env_name)
    vec_env = wrappers.Monitor(env, f"./.gym-results/{args.train_output}", force=True)
    observation = env.reset()

    #vec_env = DummyVecEnv([lambda: gym.make(train_arguments["task_name"])])
    #vec_env = VecVideoRecorder(vec_env, "./video", record_video_trigger=lambda x: x == 0, video_length=1000)
    observation = vec_env.reset()
    for i in range(1000):
        states = None
        action, _states = model.predict(observation, state=states, deterministic=True)
        observation, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
            observation = vec_env.reset()

    vec_env.close()


def dir_path(path):
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'))

    p.add_argument('--train_output',
                   help="The directory where the training output & configs were logged to",
                   type=dir_path,
                   default='runs/GNN_PPO_inp_64_pro_64324_pol_64_val_64_64_N2048_B512_lr2e-04_mode_action_per_controller_Epochs_30_Nenvs_16_GRU_AntBulletEnv-v0_10-03_23-44-46')

    p.add_argument("--num_episodes",
                   help="The number of episodes to run to evaluate the model",
                   type=int,
                   default=1)

    p.add_argument('--render',
                   help='Whether to render the evaluation with pybullet client',
                   type=bool,
                   default=False)

    p.add_argument('--save_again',
                   help='Whether to save the model in a way we can load it on any system',
                   type=bool,
                   default=False)

    args = p.parse_args()

    if args.config is not None:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list) and arg_dict[key] is not None:
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    return args


if __name__ == '__main__':
    create_video(parse_arguments())