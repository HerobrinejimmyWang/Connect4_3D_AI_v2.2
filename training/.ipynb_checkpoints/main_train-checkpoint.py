import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import random
import multiprocessing
from trainer import Trainer, TrainerArgs

# Protect entry point for multiprocessing
if __name__ == "__main__":
    # Ensure standard python multiprocessing behavior
    multiprocessing.freeze_support()
    mp.set_start_method('spawn', force=True)

    # --- Global random seed for reproducibility ---
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    
    # Define Args
    args = TrainerArgs()
    
    # --- Custom Configuration ---
    args.num_iterations = int(os.environ.get('MAIN_TRAIN_NUM_ITERATIONS', '300') or 300)          # Early formal training phase
    args.num_self_play_games = 200    # Increase this if using parallel to get better data
    args.num_channels = 256
    args.checkpoint_interval = 4
    args.eval_interval = 4
    args.lr_decay_step_size = 40      # 每 40 轮降低一次学习率
    args.lr_decay_gamma = 0.8         # 衰减系数 0.8
    args.epochs = 4
    args.learning_rate = 0.0005
    args.policy_head_lr_scale = 1.0
    args.value_head_lr_scale = 1.0
    args.max_checkpoints = 4          # Keep only 3 latest checkpoints to save space
    args.eval_games = 10              # legacy参数（单best时代）；当前双代best逻辑不再使用该值
    args.update_threshold = 0.55      # legacy参数（单best时代）；当前请使用 best_update_threshold
    args.best_eval_games_per_generation = 30   # 与每一代 best 的评估局数
    args.best_eval_required_generations = 2    # 需要同时通过几代 best（2 表示 recent + older）
    args.best_update_threshold = 0.55          # 新模型需在每个必选对手上达到该胜率
    args.best_eval_parallelize_generations = True  # recent/older 两组评估并行执行
    args.best_recent_filename = 'best_new.pth.tar'
    args.best_older_filename = 'best_old.pth.tar'
    args.best_legacy_filename = 'best.pth.tar'  # 兼容旧脚本读取路径
    args.self_play_workers = 64       # Number of parallel workers for self-play; adjust based on CPU cores and memory
    args.shared_inference_server_count = 8  # 当前模型自对弈默认启用 10 个 GPU 推理服务进程，减轻 CPU 侧排队
    args.high_mcts_shared_inference_server_threshold = 0
    args.high_mcts_shared_inference_server_count = 4
    args.compatible_inference_server_count = 1  # 教师/兼容模型默认保持单服务，避免重复占用显存
    args.num_mcts_threads = 8         # Number of threads for MCTS during self-play
    args.inference_batch_size = 128   # 提高批大小以压榨 GPU
    args.inference_timeout_s = 0.001  # 缩短等待时间，配合多服务器架构提升响应
    args.virtual_loss = 1.2           # Virtual loss for MCTS to encourage exploration in parallel settings
    args.tactical_override_max_step = 15  # 前 20 步若存在立即赢或必须防的位置，直接短路为战术动作
    args.tactical_override_prefer_win = False
    args.tactical_override_prefer_block = False
    args.dirichlet_alpha = 0.25       # 稍微提高阿尔法值，使根节点探索更广
    args.dirichlet_epsilon = 0.20     # 增加噪声比例，强制引入更多策略多样性
    args.self_play_exploration_strength = 1.0
    args.self_play_phase_schedule = [
        {
            'name': 'opening_stable',
            'max_step': 9,
            'temperature': 0.5,
            'dirichlet_alpha': 0.5,
            'dirichlet_epsilon': 0.006,
        },
        {
            'name': 'early_midgame_probe',
            'max_step': 30,
            'temperature': 1.0,
            'dirichlet_alpha': 0.24,
            'dirichlet_epsilon': 0.060,
        },
        {
            'name': 'mid_lategame_greedy',
            'max_step': 55,
            'temperature': 0.5,
            'dirichlet_alpha': 0.5,
            'dirichlet_epsilon': 0.005,
        },
        {
            'name': 'lategame_greedy',
            'max_step': None,
            'temperature': 0.0,
            'dirichlet_alpha': 0.0,
            'dirichlet_epsilon': 0.0,
        },
    ]
    args.exploration_iteration_schedule = [
        {'start_iter': 1, 'end_iter': 60, 'temperature_scale': 1.10, 'noise_scale': 1.10},
        {'start_iter': 61, 'end_iter': 120, 'temperature_scale': 1.00, 'noise_scale': 1.00},
        {'start_iter': 121, 'end_iter': 180, 'temperature_scale': 0.90, 'noise_scale': 0.90},
        {'start_iter': 181, 'end_iter': 240, 'temperature_scale': 0.85, 'noise_scale': 0.85},
        {'start_iter': 241, 'end_iter': None, 'temperature_scale': 0.80, 'noise_scale': 0.80},
    ]
    args.history_len = 18             # 更激进地淘汰旧数据，让策略更新更快落地
    args.teacher_opponent_history_len = 18  # 教师对抗样本单独保留更久，降低切回自博弈后的遗忘风险
    args.latest_data_weight = 1.0     # 暂时关闭新样本过采样，避免与长局加权叠加
    args.min_game_steps = 1          # 过滤器保持关闭（阈值=1），短局处理交由样本加权完成
    args.min_game_steps_start_iteration = 13  # 从第 13 轮开始过滤
    args.num_mcts_sims = 1024
    args.mcts_sim_candidates = [1024]
    args.mcts_promotion_improve_count = 0
    args.eval_interval_after_best = 4      # 一旦出现新 best，后续一段时间更频繁评估
    args.eval_boost_rounds_after_improve = 2
    args.random_baseline_stability_threshold = 0.60
    args.always_evaluate_random_baseline = False
    args.random_baseline_eval_min_mcts_sims = 256
    args.enable_random_baseline_eval = False
    
    args.loss_increase_patience = 300  # 关闭这种不太科学的早停，AlphaZero Loss 上涨很正常
    args.no_improve_eval_patience = 12  

    default_auxiliary_model = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'save_model (old v2.1)',
        'High',
        'best.pth.tar',
    )
    args.auxiliary_model_path = os.environ.get('AUXILIARY_MODEL_PATH', default_auxiliary_model)
    args.auxiliary_model_label = 'v2.1_high_teacher'
    args.auxiliary_eval_games = 40
    args.auxiliary_eval_interval = 4                # 仍然保留对教师的评估，用于监控是否跑偏
    # 主训练阶段不再单独做教师蒸馏/教师buffer热启动；如需教师引导，优先从 RL checkpoint 接续。
    args.teacher_bootstrap_num_games = 0
    args.teacher_bootstrap_buffer_path = None
    args.teacher_bootstrap_regenerate = False
    args.teacher_bootstrap_mcts_sims = 128
    args.teacher_bootstrap_temperature = 0.04
    args.teacher_bootstrap_dirichlet_alpha = 0.18
    args.teacher_bootstrap_dirichlet_epsilon = 0.03
    args.teacher_bootstrap_warmup_iterations = 0
    args.teacher_replay_initial_ratio = 0.0
    args.teacher_replay_final_ratio = 0.0
    args.teacher_replay_decay_iterations = 1
    args.teacher_replay_drift_threshold = -1.0
    args.teacher_replay_drift_ratio = 0.0
    args.teacher_replay_relax_start_iteration = 0
    args.teacher_replay_relax_end_iteration = 0
    args.teacher_replay_relaxed_drift_ratio = 0.0
    # 注意，原类蒸馏的启动方式这里是关闭的配置

    args.step_stagnation_window = 5                 # 连续多少轮评估没有明显提升就认为策略停滞
    args.step_stagnation_mean_tolerance = 0.01
    args.step_stagnation_variance_tolerance = 0.05

    # --- Option A: rollback + temporary value freeze + low-temperature continuation ---
    args.enable_option_a_recovery = False  # 是否启用 Option A 的整体方案
    args.option_a_policy_only_iterations = int(os.environ.get('OPTION_A_POLICY_ONLY_ITERS', '7') or 7)
    args.option_a_low_temp_min = float(os.environ.get('OPTION_A_LOW_TEMP_MIN', '0.6') or 0.6)
    args.option_a_low_temp_max = float(os.environ.get('OPTION_A_LOW_TEMP_MAX', '0.8') or 0.8)
    args.option_a_low_temp_bridge_iterations = int(os.environ.get('OPTION_A_LOW_TEMP_BRIDGE_ITERS', '5') or 5)
    args.option_a_freeze_value_prefixes = ['val_']
    args.option_a_freeze_shared_prefixes = []

    # Adaptive GPU strategy: keep self-play on CPU workers by default, train on GPU if available.
    if torch.cuda.is_available():
        args.train_device = 'cuda'
        args.infer_device = 'cpu'
        args.shared_inference_device = 'cuda'
        args.batch_size = max(args.batch_size, 512)
    else:
        allow_cpu_training = os.environ.get('ALLOW_CPU_TRAINING', '').strip().lower()
        interactive_stdin = bool(getattr(sys.stdin, 'isatty', lambda: False)())

        if allow_cpu_training in ('1', 'true', 'y', 'yes'):
            resp = 'y'
        elif interactive_stdin:
            try:
                resp = input("未检测到 NVIDIA GPU (CUDA)。是否使用 CPU 继续训练？ [y/N]: ").strip().lower()
            except Exception:
                resp = 'n'
        else:
            resp = 'y'
            print("未检测到 NVIDIA GPU (CUDA)，当前为非交互环境，自动降级为 CPU 训练。")

        if resp in ('y', 'yes'):
            args.train_device = 'cpu'
            args.infer_device = 'cpu'
            args.shared_inference_device = 'cpu'
            print("已选择使用 CPU 进行训练。")
        else:
            print("未同意使用 CPU 训练，程序终止。")
            sys.exit(0)

    
    # --- Resume Training Configuration ---
    # Set this to True to resume from the latest checkpoint automatically
    RESUME_TRAINING = True  
    
    # Path to a specific full training checkpoint if you want to resume a specific iteration.
    # Option A default target: rollback to best.pth.tar. Environment variables can still override it.
    rollback_checkpoint_path = '/root/autodl-tmp/AI_v2.2 (2)/training/checkpoints/checkpoint272/checkpoint.pth.tar'
    CUSTOM_RESUME_PATH = (
        os.environ.get('CUSTOM_RESUME_PATH')
        or os.environ.get('RESUME_CHECKPOINT_PATH')
        or rollback_checkpoint_path
    )
        
    resume_checkpoint = None    
    if RESUME_TRAINING:
        import os
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth.tar')
        best_path = os.path.join(args.checkpoint_dir, 'best.pth.tar')
        if CUSTOM_RESUME_PATH and os.path.exists(CUSTOM_RESUME_PATH):
            resume_checkpoint = CUSTOM_RESUME_PATH
            if rollback_checkpoint_path == CUSTOM_RESUME_PATH:
                print(f"Option A rollback checkpoint selected: {resume_checkpoint}")
        elif os.path.exists(latest_path):
            resume_checkpoint = latest_path
            if CUSTOM_RESUME_PATH == rollback_checkpoint_path:
                print(f"警告：未找到 rollback checkpoint，自动回退到 latest.pth.tar")
        elif os.path.exists(best_path):
            resume_checkpoint = best_path
            print("警告：未找到 latest.pth.tar，当前将退化为从 best.pth.tar 加载权重；这不会恢复训练轮次、优化器和历史状态。")
        
        if resume_checkpoint:
            print(f"Resume Training requested. Using checkpoint: {resume_checkpoint}")
        else:
            print("Resume Training requested but no checkpoint found. Starting from scratch.")
    
    # --- Check Iterations ---
    if RESUME_TRAINING and resume_checkpoint:
        # Simple check: if start_iter (from checkpoint) >= num_iterations, warn user
        import numpy as np
        try:
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')
        except Exception:
            # 备用路径（不同 numpy 版本）
            try:
                with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                    checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')
            except Exception as e:
                print("警告：safe_globals 两次尝试均失败；仅在信任 checkpoint 时使用 weights_only=False")
                checkpoint_data = torch.load(resume_checkpoint, map_location='cpu', weights_only=False) 
        
        # checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')

        if isinstance(checkpoint_data, dict) and 'iteration' in checkpoint_data:
            last_iter = checkpoint_data['iteration']
            if last_iter >= args.num_iterations:
                print(f"WARNING: Checkpoint is at iteration {last_iter}, but num_iterations is set to {args.num_iterations}.")
                print(f"Increasing num_iterations to {last_iter + 10} to continue training.")
                args.num_iterations = last_iter + 10
        del checkpoint_data # Free memory

    print("Initializing Training...")
    auxiliary_eval_enabled = (
        int(getattr(args, 'auxiliary_eval_games', 0) or 0) > 0
        and int(getattr(args, 'auxiliary_eval_interval', 0) or 0) > 0
    )
    if auxiliary_eval_enabled and args.auxiliary_model_path and os.path.exists(args.auxiliary_model_path):
        print(f"Auxiliary teacher enabled: {args.auxiliary_model_label} -> {args.auxiliary_model_path}")
    elif auxiliary_eval_enabled and args.auxiliary_model_path:
        print(f"Auxiliary teacher not found, skipping: {args.auxiliary_model_path}")
    trainer = Trainer(args, resume_path=resume_checkpoint)
    
    print("Starting Training Loop...")
    trainer.train()