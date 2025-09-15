import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import sys
from types import SimpleNamespace
from utils.keyboard_listener import KeyboardListener


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # 键盘监听器（传入带 cmd 属性的对象）
    robot_cmd_holder = SimpleNamespace(cmd=cmd)
    listener = KeyboardListener(robot_cmd_holder)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d, key_callback=listener.keyboard_callback) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # Flag to know whether we've printed the info block at least once
        previous_printed = False
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                lin_vel = d.qvel[0:3]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_for_obs = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega_for_obs
                obs[3:6] = gravity_orientation
                # cmd 可能被 listener 修改，使用 holder 中最新值
                cmd = robot_cmd_holder.cmd
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # 中文输出：以 2 列 3 行表格打印机器人速度，并打印指令速度
                v_x, v_y, v_z = lin_vel.tolist()
                w_x, w_y, w_z = omega.tolist()
                table = (
                    f"[机器人根速度]  [线速度 m/s | 角速度 rad/s]\n"
                    f" v_x:{v_x:+.3f} | w_x:{w_x:+.3f}\n"
                    f" v_y:{v_y:+.3f} | w_y:{w_y:+.3f}\n"
                    f" v_z:{v_z:+.3f} | w_z:{w_z:+.3f}"
                )
                # 指令速度中文格式：前向/后向指令速度，左/右转速度（若存在）
                # Prepare short summary values
                short_table = (
                    f"v: {lin_vel[0]:+.3f}/{lin_vel[1]:+.3f}/{lin_vel[2]:+.3f} m/s | "
                    f"w: {omega[0]:+.3f}/{omega[1]:+.3f}/{omega[2]:+.3f} rad/s"
                )

                # Build the command parts, hiding zero components and choosing labels
                cmd_parts = []
                try:
                    if len(cmd) >= 1:
                        vx = float(cmd[0])
                        if abs(vx) > 1e-9:
                            if vx >= 0:
                                cmd_parts.append(f"前向指令速度: {vx:.3f} m/s")
                            else:
                                cmd_parts.append(f"后向指令速度: {abs(vx):.3f} m/s")
                    if len(cmd) >= 2:
                        vy = float(cmd[1])
                        if abs(vy) > 1e-9:
                            if vy > 0:
                                cmd_parts.append(f"向左横移速度: {vy:.3f} m/s")
                            else:
                                cmd_parts.append(f"向右横移速度: {abs(vy):.3f} m/s")
                    if len(cmd) >= 3:
                        yaw = float(cmd[2])
                        if abs(yaw) > 1e-9:
                            if yaw > 0:
                                cmd_parts.append(f"左转速度: {yaw:.3f} rad/s")
                            else:
                                cmd_parts.append(f"右转速度: {abs(yaw):.3f} rad/s")
                except Exception:
                    cmd_list = cmd.tolist() if hasattr(cmd, 'tolist') else list(cmd)
                    cmd_parts = [str(cmd_list)]

                # Compose the multi-line block we will print and overwrite each frame.
                # We use an ANSI cursor move to go up N lines before printing the block so it overwrites the previous block.
                # Block lines (fixed number): 3 lines for cmd & header + 3 lines for velocity table = 6 lines total.
                cmd_line = "[当前指令速度] " + "，".join(cmd_parts) if cmd_parts else "[当前指令速度] 0"

                table_lines = [
                    "[机器人根速度]  [线速度 m/s | 角速度 rad/s]",
                    f" v_x:{lin_vel[0]:+.3f} | w_x:{omega[0]:+.3f}",
                    f" v_y:{lin_vel[1]:+.3f} | w_y:{omega[1]:+.3f}",
                    f" v_z:{lin_vel[2]:+.3f} | w_z:{omega[2]:+.3f}",
                ]

                # ANSI: move cursor up by previous_block_lines and carriage return. We'll track a constant height.
                block_height = 1 + len(table_lines)  # cmd line + table lines

                # Attempt to overwrite the previous block; on the first print avoid moving the cursor up
                # to prevent moving above the terminal buffer.
                if previous_printed:
                    # Move cursor to start of the block
                    sys.stdout.write(f"\r\x1b[{block_height}A")
                # Print command line and table lines, clearing each line first to avoid leftover chars
                def _clear_and_print(line: str):
                    # Clear the entire line then write the content and newline
                    sys.stdout.write('\r\x1b[2K')
                    sys.stdout.write(line + "\n")

                _clear_and_print(cmd_line)
                for l in table_lines:
                    _clear_and_print(l)
                sys.stdout.flush()
                previous_printed = True
                try:
                    parts = []
                    if len(cmd) >= 1:
                        vx = float(cmd[0])
                        if vx >= 0:
                            parts.append(f"前向指令速度: {vx:.3f} m/s")
                        else:
                            parts.append(f"后向指令速度: {abs(vx):.3f} m/s")
                    if len(cmd) >= 2:
                        vy = float(cmd[1])
                        if vy > 0:
                            parts.append(f"向左横移速度: {vy:.3f} m/s")
                        elif vy < 0:
                            parts.append(f"向右横移速度: {abs(vy):.3f} m/s")
                        else:
                            parts.append("左/右横移速度: 0.000 m/s")
                    if len(cmd) >= 3:
                        yaw = float(cmd[2])
                        if yaw > 0:
                            parts.append(f"左转速度: {yaw:.3f} rad/s")
                        elif yaw < 0:
                            parts.append(f"右转速度: {abs(yaw):.3f} rad/s")
                        else:
                            parts.append("左/右转速度: 0.000 rad/s")
                    # rebuild cmd_parts from parsed parts and print is handled above in the unified block
                    pass
                except Exception:
                    # 如果指令格式异常，降级为原始打印
                    pass

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        # When viewer loop exits, ensure the last printed line ends with a newline
        print()
