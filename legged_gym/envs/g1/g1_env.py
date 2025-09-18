
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch

class G1Robot(LeggedRobot):
    def _get_body_indices(self):
        super()._get_body_indices()
        knee_names = [name for name in self.body_names if "knee_link" in name]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(knee_names):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_height = torch.zeros(self.num_envs, self.feet_num, device=self.device)
        self.last_feet_z = self.feet_pos[:, :, 2].clone()

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.stance_mask = torch.zeros(self.num_envs, self.feet_num, dtype=torch.bool, device=self.device)
        self.swing_mask = torch.zeros_like(self.stance_mask)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        delta_z = self.feet_pos[:, :, 2] - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = self.feet_pos[:, :, 2]
        if hasattr(self, "contact_forces"):
            contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_force_threshold
        else:
            contact = torch.zeros_like(self.feet_height, dtype=torch.bool)
        self.feet_height *= ~contact

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        threshold = getattr(self.cfg.rewards, "gait_stance_threshold", 0.55)
        self.stance_mask = self.leg_phase < threshold
        self.swing_mask = ~self.stance_mask

        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    # ----- G1 专有奖励 -----
    def _reward_contact(self):
        # 支撑脚着地与步态相位匹配的奖励
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        # 摆动脚高度偏离目标时的惩罚
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # 存活奖励，防止早期终止
        return 1.0
    
    def _reward_contact_no_vel(self):
        # 足端接触但无速度时的惩罚，避免拖拽
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_hip_pos(self):
        # 抑制髋关节侧向摇摆
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    def _reward_feet_clearance(self):
        # 摆动脚达到期望抬脚高度的奖励
        if self.feet_num == 0:
            return torch.zeros(self.num_envs, device=self.device)
        swing_mask = self.swing_mask.float()
        clearance_target = getattr(self.cfg.rewards, "target_feet_height", 0.08)
        tolerance = getattr(self.cfg.rewards, "feet_clearance_tolerance", 0.01)
        reward_hit = (torch.abs(self.feet_height - clearance_target) < tolerance).float()
        return torch.sum(reward_hit * swing_mask, dim=1)

    def _reward_feet_contact_number(self):
        # 实际支撑脚数量与期望支撑脚数量一致时的奖励
        if self.feet_num == 0:
            return torch.zeros(self.num_envs, device=self.device)
        penalty = getattr(self.cfg.rewards, "contact_mismatch_penalty", 0.3)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_force_threshold
        reward = torch.where(contact == self.stance_mask, torch.ones_like(self.feet_height), torch.full_like(self.feet_height, -penalty))
        return torch.mean(reward, dim=1)

    def _reward_foot_slip(self):
        # 接触期足端滑移的惩罚
        if self.feet_num == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_force_threshold
        foot_speed_norm = torch.norm(self.feet_vel[:, :, :2], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_distance(self):
        # 双脚横向间距保持在安全范围内的奖励
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, device=self.device)
        foot_pos = self.feet_pos[:, :, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = getattr(self.cfg.rewards, "min_dist", 0.2)
        max_df = getattr(self.cfg.rewards, "max_dist", 0.5)
        decay = getattr(self.cfg.rewards, "distance_decay", 100.0)
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0., 0.5)
        return (torch.exp(-torch.abs(d_min) * decay) + torch.exp(-torch.abs(d_max) * decay)) / 2

    def _reward_knee_distance(self):
        # 膝部间距保持在期望范围内的奖励
        if not hasattr(self, "knee_indices") or self.knee_indices.numel() < 2:
            return torch.zeros(self.num_envs, device=self.device)
        knee_pos = self.rigid_body_states_view[:, self.knee_indices.long(), :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = getattr(self.cfg.rewards, "min_dist", 0.2)
        max_df = getattr(self.cfg.rewards, "max_knee_dist", 0.5)
        decay = getattr(self.cfg.rewards, "distance_decay", 100.0)
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0., 0.5)
        return (torch.exp(-torch.abs(d_min) * decay) + torch.exp(-torch.abs(d_max) * decay)) / 2

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            ids = env_ids.long()
        else:
            if len(env_ids) == 0:
                return
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self.update_feet_state()
        self.last_feet_z[ids] = self.feet_pos[ids, :, 2]
        self.feet_height[ids] = 0.
    
