import numpy as np

try:
    from mujoco.glfw import glfw
except Exception as e:  # 兼容环境无 mujoco/glfw 的情况
    glfw = None


class KeyboardListener:
    """
    8246/79/空格 控制模式：
    - 8/2: v_x 正/负档位
    - 4/6: v_y 正/负档位（左为正，右为负）
    - 7/9: yaw 正/负档位（左转为正，右转为负）
    - 5: 速度归零（档位清零）

    每次按键切换一档；相反方向按键会先将档位向 0 方向减少，归零后再累加相反方向档位。
    cmd 采用归一化值，随后在上层乘以 cmd_scale 使用。
    需要 robot.cmd 至少包含前三个通道 [vx, vy, yaw]。
    """

    def __init__(self, robot, lin_step: float = 0.1, yaw_step: float = 0.1, max_norm: float = 1.0):
        self.robot = robot
        self.lin_step = float(lin_step)
        self.yaw_step = float(yaw_step)
        self.max_norm = float(max_norm)
        # 档位计数（整数）
        self.gear_vx = 0
        self.gear_vy = 0
        self.gear_yaw = 0

    def _has_index(self, idx: int) -> bool:
        try:
            return len(self.robot.cmd) > idx
        except Exception:
            return False

    def _ensure_array(self):
        if isinstance(self.robot.cmd, list):
            self.robot.cmd = np.array(self.robot.cmd, dtype=np.float32)

    def _apply_gears(self):
        self._ensure_array()
        if not self._has_index(2):
            return
        vx = np.clip(self.gear_vx * self.lin_step, -self.max_norm, self.max_norm)
        vy = np.clip(self.gear_vy * self.lin_step, -self.max_norm, self.max_norm)
        yaw = np.clip(self.gear_yaw * self.yaw_step, -self.max_norm, self.max_norm)
        self.robot.cmd[0] = vx
        self.robot.cmd[1] = vy
        self.robot.cmd[2] = yaw

    def keyboard_callback(self, key):
        if glfw is None:
            return
        if not hasattr(self.robot, "cmd") or self.robot.cmd is None:
            return

        # 8246 前后左右
        if key == glfw.KEY_KP_8:
            self.gear_vx += 1
            self._apply_gears()
        elif key == glfw.KEY_KP_2:
            self.gear_vx -= 1
            self._apply_gears()
        elif key == glfw.KEY_KP_4:
            self.gear_vy += 1
            self._apply_gears()
        elif key == glfw.KEY_KP_6:
            self.gear_vy -= 1
            self._apply_gears()
        # 79 左右转向
        elif key == glfw.KEY_KP_7:
            self.gear_yaw += 1
            self._apply_gears()
        elif key == glfw.KEY_KP_9:
            self.gear_yaw -= 1
            self._apply_gears()
        # 5 归零
        elif key == glfw.KEY_KP_5:
            self.gear_vx = 0
            self.gear_vy = 0
            self.gear_yaw = 0
            self._apply_gears()
        else:
            # 其他按键忽略
            pass
