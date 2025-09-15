import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # enforce requested defaults: default max iterations 10000 when not provided
    try:
        if train_cfg.runner.max_iterations is None or train_cfg.runner.max_iterations <= 0:
            train_cfg.runner.max_iterations = 10000
    except Exception:
        train_cfg.runner.max_iterations = 10000
    # set save interval to 500 iterations (checkpoints every 500, skip 0)
    train_cfg.runner.save_interval = 500

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
