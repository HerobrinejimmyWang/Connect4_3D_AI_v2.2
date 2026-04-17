"""Python distillation config.

Use with:
  python distillation/main_distill.py --config distillation/distill_config.py
"""

CONFIG = {
    # -----------------------------
    # Run / checkpoint
    # -----------------------------
    "run_name": "distill_v1.2",
    "checkpoint_root": "distillation/checkpoints",
    "num_iterations": 200,
    "checkpoint_interval": 4,
    "eval_interval": 4,
    "batch_size": 256,
    "epochs": 4,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    # Table-driven LR decay (preferred over fixed step_size/gamma).
    # Supports either absolute "lr" or relative "lr_scale".
    "learning_rate_schedule": [
        {"start_iter": 1, "end_iter": 12, "lr_scale": 1.00},
        {"start_iter": 13, "end_iter": 50, "lr_scale": 0.60},
        {"start_iter": 51, "end_iter": 100, "lr_scale": 0.48},
        {"start_iter": 101, "end_iter": 160, "lr_scale": 0.384},
        {"start_iter": 161, "end_iter": None, "lr_scale": 0.3072},
    ],

    # -----------------------------
    # Student model preset
    # -----------------------------
    "active_model_preset": "fast",
    "balanced_model_preset": {
        "num_channels": 224,
        "num_res_blocks": 4,
    },
    "fast_model_preset": {
        "num_channels": 96,
        "num_res_blocks": 3,
    },

    # -----------------------------
    # Models
    # -----------------------------
    "teacher_model_path": "/root/autodl-tmp/AI_v2.2 (2)/save_model/v2.2_balence_Iter188/best_recent.pth.tar",
    "v21_high_model_path": "/root/autodl-tmp/AI_v2.2 (2)/save_model (old v2.1)/High/best.pth.tar",

    # -----------------------------
    # Hot start / teacher cache
    # -----------------------------
    "hot_start_iterations": 4,
    "hot_start_teacher_games_per_iteration": 250,
    "teacher_data_cache_path": "distillation/cache/teacher_examples.pth.tar",
    "teacher_data_generation_enabled": True,
    "teacher_data_generate_mode": False,
    "teacher_data_generation_games": 1000,

    # Legacy fallback keys (still supported)
    "teacher_num_mcts_sims": 1600,
    "teacher_label_temperature": 0.5,

    # -----------------------------
    # Distillation loss / mix
    # -----------------------------
    "num_self_play_games": 128,
    # History window (iteration-level) to mitigate catastrophic forgetting.
    # Per-source values <= 0 will fall back to history_window_len.
    "history_window_len": 12,
    "self_play_history_window_len": 12,
    "adversarial_history_window_len": 16,
    "teacher_history_window_len": 16,

    "policy_loss_weight": 1.0,
    "value_loss_weight": 1.0,
    "self_play_loss_weight": 1.0,
    "adversarial_loss_teacher_weight": 1.0,
    "adversarial_win_student_weight": 0.35,
    "adversarial_win_teacher_response_weight": 0.35,
    "teacher_loss_decay_start": 4,
    "teacher_loss_decay_end": 12,
    "teacher_loss_floor": 0.05,

    # Schedule can start from iter 5 because iter 1-4 are hot-start only.
    "teacher_mix_schedule": [
        {"start_iter": 5, "end_iter": 12, "teacher_ratio": 0.65},
        {"start_iter": 13, "end_iter": 20, "teacher_ratio": 0.20},
        {"start_iter": 21, "end_iter": None, "teacher_ratio": 0.0},
    ],

    # Legacy fallback schedules (used before further-train exploration kicks in)
    "exploration_temperature_schedule": [
        {"start_iter": 5, "end_iter": 12, "temperature": 0.6},
    ],
    "self_play_mcts_schedule": [
        (1, None, 512),
    ],

    # -----------------------------
    # Search profile split
    # -----------------------------
    # Student self-play thinking strength.
    "student_self_play_search": {
        "mcts_schedule": [
            (1, None, 512),
        ],
        # Use legacy fixed-temperature mode before this iteration; switch to
        # AlphaZero-style staged exploration after this iteration.
        "further_train_start_iter": 13,
        "self_play_phase_schedule": [
            {
                "name": "opening_stable",
                "max_step": 8,
                "temperature": 0.5,
                "dirichlet_alpha": 0.5,
                "dirichlet_epsilon": 0.006,
            },
            {
                "name": "early_midgame_probe",
                "max_step": 28,
                "temperature": 1.0,
                "dirichlet_alpha": 0.24,
                "dirichlet_epsilon": 0.060,
            },
            {
                "name": "mid_lategame_greedy",
                "max_step": 50,
                "temperature": 0.5,
                "dirichlet_alpha": 0.5,
                "dirichlet_epsilon": 0.005,
            },
            {
                "name": "lategame_greedy",
                "max_step": None,
                "temperature": 0.0,
                "dirichlet_alpha": 0.0,
                "dirichlet_epsilon": 0.0,
            },
        ],
        "exploration_iteration_schedule": [
            {"start_iter": 13, "end_iter": 50, "temperature_scale": 1.0, "noise_scale": 1.0},
            {"start_iter": 51, "end_iter": 100, "temperature_scale": 0.9, "noise_scale": 0.9},
            {"start_iter": 101, "end_iter": None, "temperature_scale": 0.8, "noise_scale": 0.8},
        ],
        "cpuct": 1.0,
        "num_mcts_threads": 4,
        "virtual_loss": 1.0,
        "dirichlet_alpha": 0.30,
        "dirichlet_epsilon": 0.25,
        "self_play_exploration_strength": 1.0,
        "inference_batch_size": 64,
        "inference_timeout_s": 0.003,
        "shared_inference_server_count": 4,
        "compatible_inference_server_count": 4,
        "high_mcts_shared_inference_server_threshold": 1024,
        "high_mcts_shared_inference_server_count": 6,
    },

    # Teacher cache generation thinking strength.
    "teacher_cache_search": {
        "num_mcts_sims": 1600,
        "label_temperature": 0.5,
        "noise_scale": 0.2,
        "cpuct": 1.0,
        "num_mcts_threads": 8,
        "virtual_loss": 1.2,
        "dirichlet_alpha": 0.30,
        "dirichlet_epsilon": 0.25,
        "inference_batch_size": 128,
        "inference_timeout_s": 0.001,
        "compatible_inference_server_count": 4,
        "shared_inference_server_count": 6,
        "high_mcts_shared_inference_server_threshold": 1024,
        "high_mcts_shared_inference_server_count": 6,
    },

    # Student-vs-teacher adversarial collection strength.
    "teacher_adversarial_search": {
        # Student side search (model)
        "model_mcts_schedule": [
            (5, None, 512),
        ],
        "model_cpuct": 1.0,
        "model_num_mcts_threads": 4,
        "model_virtual_loss": 1.0,

        # Teacher side search
        "teacher_num_mcts_sims": 512,
        "teacher_cpuct": 1.0,
        "teacher_num_mcts_threads": 4,
        "teacher_virtual_loss": 1.0,

        # Temperature control for adversarial data collection
        "game_temperature_schedule": [
            {"start_iter": 5, "end_iter": 12, "temperature": 0.5},
            {"start_iter": 13, "end_iter": None, "temperature": 1.0},
        ],
        "target_temperature": 0.5,
        "student_target_temperature": 0.5,
        "teacher_response_temperature": 0.5,

        # Optional noise in adversarial game collection
        "noise_scale": 0.0,
        "dirichlet_alpha": 0.30,
        "dirichlet_epsilon": 0.25,

        # Dual inference servers for adversarial stage
        "model_inference_batch_size": 128,
        "model_inference_timeout_s": 0.001,
        "model_inference_server_count": 4,
        "teacher_inference_batch_size": 128,
        "teacher_inference_timeout_s": 0.001,
        "teacher_inference_server_count": 4,
    },

    # -----------------------------
    # Evaluation / best refresh
    # -----------------------------
    "eval_mcts_sims": 512,
    "best_eval_games_per_generation": 30,
    "best_eval_required_generations": 2,
    "best_eval_parallelize_generations": True,
    "baseline_eval_parallelize": True,
    "best_update_threshold": 0.55,
    "eval_games_vs_teacher": 30,
    "eval_games_vs_v21_high": 30,
    "enable_best_refresh": True,
    "info_log_name": "train_info.log",

    # -----------------------------
    # Resume / rollback
    # -----------------------------
    "resume": False,
    "resume_path": None,
    "resume_weights_only": False,
    "rollback_iteration": None,
    "continue_from_iteration": None,
    "restore_optimizer_state": True,
    "restore_schedule_state": True,
    "force_overwrite": False,

    # -----------------------------
    # Runtime / workers
    # -----------------------------
    "train_device": "cuda",
    "shared_inference_device": "cuda",
    "self_play_workers": 64,
    "max_self_play_workers": 64,
    "self_play_cpu_ratio": 0.75,
}
