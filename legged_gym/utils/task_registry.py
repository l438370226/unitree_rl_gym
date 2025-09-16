import os
from datetime import datetime
import importlib
import inspect
import shutil
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        # Save environment configuration and environment source files alongside logs
        try:
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                src_dump_dir = os.path.join(log_dir, "src_dump")
                os.makedirs(src_dump_dir, exist_ok=True)

                # Copy parent/base env config (LeggedRobot)
                from legged_gym.envs.base import legged_robot_config as parent_cfg_mod
                parent_cfg_src = parent_cfg_mod.__file__
                parent_cfg_dst = os.path.join(src_dump_dir, os.path.basename(parent_cfg_src))
                if os.path.abspath(parent_cfg_src) != os.path.abspath(parent_cfg_dst):
                    shutil.copyfile(parent_cfg_src, parent_cfg_dst)

                # Copy child/current task config (e.g., g1_config.py)
                cfg_cls = env.cfg.__class__
                child_cfg_mod = importlib.import_module(cfg_cls.__module__)
                child_cfg_src = inspect.getsourcefile(child_cfg_mod) or getattr(child_cfg_mod, "__file__", None)
                if child_cfg_src is not None:
                    child_cfg_dst = os.path.join(src_dump_dir, os.path.basename(child_cfg_src))
                    if os.path.abspath(child_cfg_src) != os.path.abspath(child_cfg_dst):
                        shutil.copyfile(child_cfg_src, child_cfg_dst)

                # Copy parent/base env (LeggedRobot)
                from legged_gym.envs.base import legged_robot as parent_env_mod
                parent_env_src = parent_env_mod.__file__
                parent_env_dst = os.path.join(src_dump_dir, os.path.basename(parent_env_src))
                if os.path.abspath(parent_env_src) != os.path.abspath(parent_env_dst):
                    shutil.copyfile(parent_env_src, parent_env_dst)

                # Copy child/current task env (e.g., g1_env.py)
                child_env_mod = importlib.import_module(env.__class__.__module__)
                child_env_src = inspect.getsourcefile(child_env_mod) or getattr(child_env_mod, "__file__", None)
                if child_env_src is not None:
                    child_env_dst = os.path.join(src_dump_dir, os.path.basename(child_env_src))
                    if os.path.abspath(child_env_src) != os.path.abspath(child_env_dst):
                        shutil.copyfile(child_env_src, child_env_dst)
        except Exception as e:
            print(f"[WARN] Failed to snapshot env/config source files: {e}")

        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
