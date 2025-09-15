Quick start scripts for unitree_rl_gym

This folder contains convenience shell scripts to quickly launch training, playback, and MuJoCo deployment.

Files
- start_train.sh: Launch `legged_gym/scripts/train.py` with common flags. Usage:
  ./scripts/start_train.sh --task g1 --rl_device cuda:0 --num_envs 64 --experiment_name myexp

- start_play.sh: Run `legged_gym/scripts/play.py` (exports policy by default). Usage:
  ./scripts/start_play.sh --task g1 --rl_device cuda:0 [--headless]

- start_mujoco.sh: Run deploy MuJoCo viewer using a config from `deploy/deploy_mujoco/configs`.
  Example:
  ./scripts/start_mujoco.sh g1.yaml

Make scripts executable
Run the following once to make scripts executable:

  chmod +x scripts/*.sh

Notes
- These scripts are small wrappers and forward unknown arguments to the underlying Python scripts.
- Ensure your Python environment has required packages (Isaac Gym / MuJoCo / torch) installed and that `python` in PATH points to the desired interpreter.
