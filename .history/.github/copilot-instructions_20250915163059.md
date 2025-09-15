This repository contains RL training, simulation (Isaac Gym + MuJoCo) and real-robot deployment code for Unitree robots (go2, g1, h1, h1_2).

Key goals for AI coding agents working on this repo:
- Understand the three main flows: training (legged_gym/scripts/train.py), playback/export (legged_gym/scripts/play.py) and deployment (deploy/deploy_mujoco, deploy/deploy_real).
- Preserve existing project conventions: configuration objects under `legged_gym/envs/*_config.py`, BaseConfig auto-instantiates nested classes, and exported policies live under `logs/.../exported/policies` or `deploy/pre_train`.

Essential files/dirs to inspect when changing behavior:
- `legged_gym/` — core env/task implementations. See `legged_gym/envs/base/base_config.py` and `legged_gym/envs/base/base_task.py` for common patterns (device selection, buffer shapes, viewer handling).
- `legged_gym/scripts/{train.py,play.py}` — CLI entry points; respect their CLI flags (task, headless, resume, checkpoint).
- `deploy/deploy_mujoco/` — MuJoCo deployment harness. `deploy_mujoco.py` reads YAML configs in `configs/` and expects exported TorchScript policies.
- `deploy/deploy_real/` — real-robot deploy helpers and `cpp_g1` example (uses LibTorch).

Conventions and gotchas (use these precisely):
- Config objects: many modules expect a `cfg` following the BaseConfig convention (nested classes auto-instantiated). When editing configs, follow their nested class style rather than plain dicts.
- Devices: code distinguishes `sim_device` (e.g., `cuda:0` or `cpu`) and `rl_device`. Follow existing device string parsing when adding flags.
- Policy formats: Play exports networks as `policy_1.pt` (MLP) or `policy_lstm_1.pt` (RNN). `deploy_mujoco` expects a TorchScript model loaded with `torch.jit.load` — ensure exported artifacts are compatible.
- Observer/action ordering: Observation vectors are concatenated manually in several places (see `deploy_mujoco.py` and envs). When changing obs/action layouts, update all places that index into them.

Common developer workflows (commands visible from repo):
- Train: `python legged_gym/scripts/train.py --task=<go2|g1|h1|h1_2> [--headless true] ...`
- Play: `python legged_gym/scripts/play.py --task=<...>` (also exports policies to `logs/.../exported/policies`)
- MuJoCo sim2sim: `python deploy/deploy_mujoco/deploy_mujoco.py <config.yaml>` (configs in `deploy/deploy_mujoco/configs/`)
- Real deploy (python): `python deploy/deploy_real/deploy_real.py <net_interface> <config.yaml>`
- C++ deploy example: build under `deploy/deploy_real/cpp_g1` with LibTorch; follow README instructions.

Testing & quick checks:
- There are no formal unit tests in the repo; prefer running small smoke runs (play mode or MuJoCo deploy with very short `simulation_duration`) when changing runtime code.
- For edits to JIT-exported policy input shapes, run `play.py` to export and then run `deploy/deploy_mujoco` with that exported file to confirm inference works.

Examples to cite in PRs/issues:
- If changing observation layout, reference `deploy/deploy_mujoco/deploy_mujoco.py` lines where `obs` is constructed (omega, gravity_orientation, cmd, qj, dqj, action, phase).
- If changing viewer/keyboard behavior, reference `legged_gym/envs/base/base_task.py` keyboard subscription and `deploy/deploy_mujoco/utils/keyboard_listener.py` (keyboard listener pattern).

If anything here is unclear or you want more details (build steps, device flags, or policy export examples), tell me which area to expand and I will update this file.
