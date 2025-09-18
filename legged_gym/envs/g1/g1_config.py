from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        contact_force_threshold = 1.0  # 足端接触判定阈值
        gait_stance_threshold = 0.55  # 步态支撑相相位阈值
        feet_clearance_tolerance = 0.01  # 抬脚高度允许误差
        distance_decay = 100.0  # 距离惩罚衰减系数
        min_dist = 0.2  # 双脚/膝最小期望间距
        max_dist = 0.5  # 双脚最大期望间距
        max_knee_dist = 0.5  # 膝部最大期望间距
        target_feet_height = 0.08  # 抬脚目标高度
        contact_mismatch_penalty = 0.3  # 支撑脚数量不匹配惩罚
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # 父类奖励（保持 LeggedRobot 顺序）
            tracking_lin_vel = 1.0  # 线速度追踪奖励
            tracking_ang_vel = 0.5  # 航向角速度追踪奖励
            lin_vel_z = -2.0  # 垂直速度惩罚
            ang_vel_xy = -0.05  # 横滚俯仰角速度惩罚
            orientation = -1.0  # 姿态保持惩罚
            base_height = -10.0  # 机身高度稳定性惩罚
            torques = -1.0e-5  # 力矩能量惩罚
            dof_vel = -1e-3  # 关节速度惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚
            feet_air_time = 0.0  # 腾空时间奖励（禁用）
            collision = 0.0  # 碰撞惩罚（禁用）
            action_rate = -0.01  # 动作变化惩罚
            dof_pos_limits = -5.0  # 关节限位惩罚
            stand_still = -8.0  # 零指令时静止惩罚

            # 子类奖励（G1Robot 专有，实现见 g1_env.py）
            alive = 0.15  # 存活奖励
            contact = 0.18  # 支撑脚匹配奖励
            contact_no_vel = -0.2  # 拖拽惩罚
            feet_swing_height = -20.0  # 摆动脚高度惩罚
            hip_pos = -1.0  # 髋部姿态惩罚
            feet_clearance = 1.0  # 抬脚高度奖励
            feet_contact_number = 1.2  # 支撑脚数量奖励
            foot_slip = -0.1  # 足端滑移惩罚
            feet_distance = 0.2  # 双脚间距奖励
            knee_distance = 0.2  # 膝部间距奖励

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
