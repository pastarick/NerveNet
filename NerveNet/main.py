import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from NerveNet.models.nerve_net_conv import NerveNetConv

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.policies.register_policies import policy_aliases

basepath = Path(os.getcwd())
print(basepath)
# make sure your working directory is the repository root.
if basepath.name != "tum-adlr-ws21-04":
    os.chdir(basepath.parent)

basepath = Path(os.getcwd())


graph_logs_dir = basepath / "graph_logs_new"

task_name = 'AntBulletEnv-v0'
nervenet_assets_dir = Path(os.getcwd()).parent / \
    "NerveNet" / "environments" / "assets"


env = gym.make(task_name)

log_name = '{}_{}'.format(task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
checkpoint_callback = CheckpointCallback(save_freq=50, save_path='runs/' + log_name,
                                         name_prefix='rl_model')

PPO.policy_aliases.update(policy_aliases)
A2C.policy_aliases.update(policy_aliases)

model = A2C("GnnPolicy",  # the key in the dict `policy_aliases`
            env,  # the gym environment. It is possibly created (if just a string) and it is used by the agent to
            # retrieve observation space and action space
            verbose=1,
            policy_kwargs={  # almost entirely passed to the policy class
                'mlp_extractor_kwargs': {  # they go to policy class, which passes them to class NerveNetGNN
                    'task_name': task_name,  # the name of the task, in theory it should be from a pool of tasks
                    # already present in pybullet
                    # as an alternative to attr `task_name` one can specify `xml_name` which is directly the name of xml
                    'xml_assets_path': None,  # if None, the default behavior is to look for the xml files in the
                    # pybullet folder
                },
                'net_arch':  {  # they go to policy class, which passes them to ActorCriticPolicy from sb3
                    "input": [
                        (nn.Linear, 6),
                    ],
                    "propagate": [
                        (NerveNetConv, 12),
                        # (NerveNetConv, 12),
                        # (NerveNetConv, 12)
                    ],
                    "policy": [
                        (nn.Linear, 64),
                        (nn.Linear, 64)
                    ],
                    "value": [
                        (nn.Linear, 64),
                        (nn.Linear, 1)
                    ]
                },
                "activation_fn":  nn.Tanh,  # go to policy class
            },
            tensorboard_log="runs",
            learning_rate=3.0e-4,
            # batch_size=64,
            # n_epochs=10,
            n_steps=1)

model.learn(total_timesteps=1000000, callback=checkpoint_callback,
            tb_log_name=log_name)
model.save("a2c_ant")
