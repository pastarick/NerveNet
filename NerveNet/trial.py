import os
from datetime import datetime
from pathlib import Path

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn

from NerveNet.models.nerve_net_conv import NerveNetConv
from NerveNet.policies.register_policies import policy_aliases

basepath = Path(os.getcwd())
print(basepath)
# make sure your working directory is the repository root.
if basepath.name != "NerveNet":
    os.chdir(basepath.parent)

basepath = Path(os.getcwd())

graph_logs_dir = basepath / "graph_logs_new"

task_name = 'HopperBulletEnv-v0'
nervenet_assets_dir = Path(os.getcwd()).parent / \
                      "NerveNet" / "environments" / "assets"

env = gym.make(task_name)

log_name = '{}_{}'.format(task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
checkpoint_callback = CheckpointCallback(save_freq=50, save_path='runs/' + log_name,
                                         name_prefix='rl_model')

PPO.policy_aliases.update(policy_aliases)
A2C.policy_aliases.update(policy_aliases)

model = PPO("GnnPolicy",
            env,
            verbose=1,
            policy_kwargs={
                'mlp_extractor_kwargs': {
                    'task_name': task_name,
                    'xml_assets_path': None,
                },
                'net_arch': {
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
                "activation_fn": nn.Tanh,
            },
            tensorboard_log="runs",
            learning_rate=3.0e-4,
            # batch_size=64,
            # n_epochs=10,
            # n_steps=1
            )

model.learn(total_timesteps=1000, callback=checkpoint_callback,
            tb_log_name=log_name)

# env = gym.make('trunk-v0')
env.seed(42)
observation = env.reset()

done = False
while not done:
    # action = env.action_space.sample()  # this is where you would insert your policy
    action, _ = model.predict(observation)
    observation, reward, done, info = env.step(action)
    env.render()

env.close()
