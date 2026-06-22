## Goal
实现一个 1k-5k 左右参数量的模型，使用人工设计的特征来构建 策略 和 Loss function ，最终实现训练一个无论什么设备（包括手机处理器）都可以运行的瞬间响应模型

（以下为我的初步构思）
## Model
**初步设计**：一个小型 MLP，输入层为手工设计的棋局特征，人为初始化第一层的权重

**Police**：Input Layer (dim: 40-100) → full connect layer 1 → Sigmoid → linear layer 2 → Sigmoid → linear layer 3 → Softmax → Output Layer

## Features
1. 棋盘位置 (Position Value)：在 5×5×6 的有重力空间棋盘中，依据对称性，每层有 6 类位置（(row, column): (1,1)(1,2)(1,3)(2,2)(2,3)(3,3)），共 6 层，可有 36 个特征。（由于是有界空间四子棋，空间价值应该是多个沙漏型的叠加，即双四棱锥）
2. 当前状态 (State Value)：棋局有一些显然的连线状态，如：a.平面：连二双边自由，连二单边自由，连三单边自由，连三双边自由（此时必胜已经出现），‘21’（一边是连二，一边单子，中间空了一个位置）; b.竖向：连二，连三（竖向固定只能向上发展，相应价值较低）； c.空间斜向…… 这些状态与近期 1-2 步密切相关，可能涉及必须防守或直接获胜的策略。
3. 未来状态 (Future Value)：一些棋局中目前不容易发现的状态，但是可能与未来构建获胜策略有关，一般涉及三维空间的状态，如（下用坐标表示，(row, column, layer)）：a.(2,1,1)+(2,2,2) 可以与 (2,4,4)联动，即使目前 (2,3,3) 没有落子； b.已有:(2,2,3)+(2,3,3)+(2,4,3)，但是竖列(2,1,z)和(2,5,z)都没有落子，此时(2,1,1)和(2,5,1)可能会压迫对方行动，因为(2,1,2)和(2,5,2)一旦对方落子，可以立刻获胜，同理，(2,1,2)和(2,5,2)也产生了相应的价值……

## 训练方式
我希望用我（人类）与模型的对抗来让模型学习参数和强化策略，最后能学习到我的一半以上策略就好。可以使用已有的人机对抗测试模块，以胜方的视角学习，每次完成对局后训练大概 10-20 epochs，然后用新的模型继续对弈。

## 已有条件
一个可以在 CPU 上以 mcts_sims = 512 思考量的情况下，平均响应步时为 3s 的 v2.2_fast 和 响应步时为 9s 的 v2.2_balence 的 2 个 AlphaZero 模型，可以引用其价值网络参与训练。这两个模型都能战胜我，只是稳定性上有区别

## 部分注意
1. 合法动作空间是 25 个，输出层只用 25 维；或者构建价值函数，把 25 的位置的落子可能遍历一遍，找一个价值最高的落子


## Implementation v0 (已完成)

已按当前计划完成一套可运行的基础代码骨架，核心是“每步对 25 个候选位置打分”的 tiny policy 训练闭环。

### 新增文件
1. `train_features/feature_extractor.py`
	- `CandidateFeatureExtractor`：提取全局特征 + 25 候选点特征。
	- 输出字段：`global`、`candidate`、`valid_mask`、`candidate_action_map`。
2. `train_features/tiny_policy_model.py`
	- `TinyCandidatePolicyNet`：共享候选打分器。
	- 当前默认结构参数量约 2001（在 1k-5k 目标区间内）。
3. `train_features/history_dataset.py`
	- `HumanHistoryDataset`：把 `test/history/*.json` 转为监督学习样本。
	- 支持胜负样本权重（winner/loser/draw）。
4. `train_features/tiny_trainer.py`
	- `TinyPolicyTrainer`：基础训练器。
	- 损失：`masked CE + optional teacher KL`（teacher 入口已预留）。
5. `train_features/main_tiny_train.py`
	- 命令行训练入口。
6. `train_features/__init__.py`
	- 对外导出核心类。

### 快速运行

在项目根目录运行：

```bash
python train_features/main_tiny_train.py --history test/history/*.json --epochs 1 --batch-size 32 --output train_features/checkpoints/tiny_policy_smoke.pth
```

### 当前烟雾测试结果
1. 样本数量：152
2. 参数量：2001
3. 已生成模型：`train_features/checkpoints/tiny_policy_smoke.pth`
4. 已生成训练日志：`train_features/checkpoints/tiny_policy_smoke.pth.metrics.json`

### 下一步（建议）
1. 接入 teacher 软标签生成（来自 `v2.2_fast`）并启用 KL 蒸馏。
2. 将 tiny 模型接入评测入口（先 human_eval，再 arena agent）。
3. 做特征消融和权重敏感性测试，保留高收益特征以压缩推理耗时。

## KL蒸馏 + 评测入口（已接入）

### 训练入口拆分（已完成）

为提高训练可持续性，已拆分为两个独立入口：
1. 人类对抗样本入口：`train_features/main_tiny_train_human.py`
2. AlphaZero 自对弈蒸馏入口：`train_features/main_tiny_train_teacher.py`

并保留了组合入口：`train_features/main_tiny_train.py`（可同时混合人类与teacher样本）。

### Resume 连续训练（已完成）

三个入口都支持 `--resume`，可从已有 tiny checkpoint 继续训练，便于累计更多样本、迭代增强模型能力。

示例（human 入口）：

```bash
python train_features/main_tiny_train_human.py \
	--history test/history/*.json \
	--resume train_features/checkpoints/tiny_policy_human_v1.pth \
	--epochs 10 \
	--output train_features/checkpoints/tiny_policy_human_v2.pth
```

示例（teacher 入口）：

```bash
python train_features/main_tiny_train_teacher.py \
	--teacher-model save_model/v2.2_fast/model.pth \
	--teacher-games 200 \
	--teacher-mcts-sims 256 \
	--teacher-cpu-worker-ratio 0.5 \
	--resume train_features/checkpoints/tiny_policy_teacher_v1.pth \
	--epochs 20 \
	--output train_features/checkpoints/tiny_policy_teacher_v2.pth
```

### Teacher 自对弈并行（已完成）

teacher 自对弈样本生成支持并行，默认在 CPU 下使用约 50% 的线程预算：
1. `--teacher-cpu-worker-ratio 0.5`：CPU线程预算比例（默认 0.5）
2. `--teacher-parallel-workers 0`：worker 数自动（默认）
3. 自动模式会根据预算分配 worker 数，并限制每个 worker 的 MCTS 线程，避免过度占用 CPU
4. 使用 GPU teacher 时默认退回单 worker，避免多进程重复占用 GPU 显存

### 1) KL蒸馏：v2.2_fast 自对弈生成样本 + 价值标签

已在 `train_features/main_tiny_train.py` 中接入 teacher 自对弈蒸馏参数，
并在 `train_features/teacher_distill.py` 中实现：
1. 加载 teacher（兼容 v2.x checkpoint）
2. teacher self-play 生成候选策略软标签
3. 生成 teacher value 标签
4. 与人机历史样本混合训练 tiny 模型

核心训练命令示例：

```bash
python train_features/main_tiny_train.py \
	--history test/history/*.json \
	--teacher-model save_model/v2.2_fast/model.pth \
	--teacher-games 20 \
	--teacher-max-steps 120 \
	--teacher-temp 0.8 \
	--teacher-kl-weight 0.25 \
	--value-loss-weight 0.2 \
	--epochs 10 \
	--batch-size 128 \
	--output train_features/checkpoints/tiny_policy_distill_v1.pth
```

备注：
1. `teacher-games` 决定 teacher 自对弈样本规模。
2. `teacher-kl-weight` 控制策略蒸馏强度。
3. `value-loss-weight` 控制 value 辅助蒸馏强度。

### 1.1 直接复用蒸馏缓存样本（推荐）

如果已有 `distillation/cache/teacher_examples.pth.tar`，可直接使用该缓存训练 tiny，跳过慢速 teacher 自对弈。

teacher 入口默认已切到 cache 模式：

```bash
python train_features/main_tiny_train_teacher.py \
	--teacher-data-source cache \
	--teacher-cache-path distillation/cache/teacher_examples.pth.tar \
	--teacher-cache-max-samples 0 \
	--epochs 20 \
	--output train_features/checkpoints/tiny_policy_teacher_cache_v1.pth
```

说明：
1. `teacher-cache-max-samples=0` 表示使用缓存中的全部样本。
2. 如果希望先小规模试跑，可设为 64/128/512。
3. 需要回退到实时自对弈时，把 `--teacher-data-source` 改成 `self-play` 或 `auto`。

### 2) 评测入口：arena + test

#### arena 入口

已支持 tiny agent。

1. `arena/agent.py` 新增 `TinyPolicyAgent`。
2. `arena/main_arena.py` 新增 `--p1-agent-type/--p2-agent-type`，支持 `auto|mcts|tiny`。
3. `auto` 模式下会自动识别 tiny checkpoint。

示例：

```bash
python arena/main_arena.py \
	--no-ui \
	--games 20 \
	--parallel-games 2 \
	--p1-model train_features/checkpoints/tiny_policy_distill_v1.pth \
	--p1-agent-type auto \
	--p1-name tiny_distill \
	--p2-model save_model/v2.2_fast/model.pth \
	--p2-agent-type mcts \
	--p2-name v2.2_fast
```

#### test 入口（human_eval）

已支持 tiny agent 自动识别。

1. `test/human_eval_config.py` 新增 `agent_type`（`auto|mcts|tiny`）。
2. `test/main_human_eval.py` 在构建 agent 时支持 tiny checkpoint 自动切换。

配置示例：

```python
EVAL_CONFIG = HumanEvalConfig(
		model_path=WORKSPACE_ROOT / "train_features" / "checkpoints" / "tiny_policy_distill_v1.pth",
		model_name="tiny_policy_distill_v1",
		agent_type="auto",
		human_name="Developer",
		human_plays_first=True,
)
```