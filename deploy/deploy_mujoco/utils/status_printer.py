"""Status printer for deploy_mujoco: renders a fixed multi-line status block and overwrites it each update.

Usage:
    from utils.status_printer import StatusPrinter
    sp = StatusPrinter()
    sp.render(cmd, lin_vel, omega)

This module uses ANSI control sequences (\x1b) to move the cursor and clear lines. It degrades
gracefully if the terminal does not support them (you'll see repeated prints instead of in-place updates).
"""
import sys
from typing import Sequence


class StatusPrinter:
    def __init__(self) -> None:
        self.previous_printed = False

    def render(self, cmd: Sequence, lin_vel, omega) -> None:
        """Render the status block.

        cmd: array-like with up to 3 elements [vx, vy, yaw]
        lin_vel: sequence of 3 floats (v_x, v_y, v_z)
        omega: sequence of 3 floats (w_x, w_y, w_z)
        """
        # Build command parts, hide near-zero components and choose labels
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
            cmd_list = cmd.tolist() if hasattr(cmd, "tolist") else list(cmd)
            cmd_parts = [str(cmd_list)]

        # Compose lines
        cmd_line = "[当前指令速度] " + "，".join(cmd_parts) if cmd_parts else ""

        table_lines = [
            "[机器人根速度]  [线速度 m/s | 角速度 rad/s]",
            f" v_x:{lin_vel[0]:+.3f} | w_x:{omega[0]:+.3f}",
            f" v_y:{lin_vel[1]:+.3f} | w_y:{omega[1]:+.3f}",
            f" v_z:{lin_vel[2]:+.3f} | w_z:{omega[2]:+.3f}",
        ]

        block_height = 1 + len(table_lines)

        # Move cursor up to overwrite previous block if we've printed before
        if self.previous_printed:
            try:
                sys.stdout.write(f"\r\x1b[{block_height}A")
            except Exception:
                # ignore errors and fallback to plain prints
                pass

        def _clear_and_print(line: str):
            # Clear the entire line then write the content and newline
            try:
                sys.stdout.write('\r\x1b[2K')
            except Exception:
                pass
            sys.stdout.write(line + "\n")

        _clear_and_print(cmd_line)
        for l in table_lines:
            _clear_and_print(l)

        # Clear an extra trailing line to remove any leftover wrapped text
        try:
            sys.stdout.write('\r\x1b[2K')
        except Exception:
            pass
        sys.stdout.flush()
        self.previous_printed = True
