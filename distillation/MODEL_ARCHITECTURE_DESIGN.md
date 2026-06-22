# 3D Connect4 分级模型架构设计

## 1. 目标

为同一套 6×5×5 重力四子棋规则设计 `mini`、`balanced`、`flagship` 三档模型。三档模型不是简单同比例增加普通 3D 残差块，而是共享相同的重力感知表示、25 列策略空间和部署接口，再按延迟预算增加主干容量与全局建模能力。

设计目标：

- 在固定响应时间而非仅固定 MCTS 次数下提高棋力。
- mini 尽量提高单位时间可完成的 MCTS 模拟数。
- balanced 作为训练稳定性、棋力和推理成本的默认平衡点。
- flagship 提高单次局面评估质量，但避免无约束地扩大普通 Conv3d 主干。
- 新旧模型可同时加载，现有 150 动作 MCTS 接口和 checkpoint v2 恢复接口保持兼容。
- 所有档位均支持作为 teacher 或 student，不要求师生架构一致。

非目标：

- 第一版不修改棋盘规则、MCTS 的 PUCT 公式或训练样本语义。
- 第一版不使用大型通用 Transformer 替代完整卷积主干。
- 未经过固定延迟棋力实验前，不删除当前 `Connect4Net`。

## 2. 当前基线

当前模型使用两通道 6×5×5 输入、普通 3×3×3 Conv3d 残差主干、150 维全连接策略头和展平全连接值头。

| 模型 | 配置 | 总参数 | 残差主干 | 策略头 | 值头 |
|---|---:|---:|---:|---:|---:|
| v2.2_fast | 96 channels × 3 blocks | 2,535,488 | 1,495,878 | 723,383 | 310,562 |
| v2.2_balence | 224 channels × 4 blocks | 11,902,338 | 10,846,984 | 727,479 | 314,658 |
| v2.2_large | 256 channels × 8 blocks | 29,391,338 | 28,332,048 | 728,503 | 315,682 |

当前缩放方式存在两个问题：

1. large 的容量几乎全部增长在 `C² × block_count` 的普通 3D 卷积主干上。
2. fast 的策略头和值头没有随主干缩小，两个头合计约占总参数的 41%，小模型的参数利用率偏低。

训练中观察到的波动不能直接全部归类为过拟合。需要区分：

- 经典过拟合：训练损失继续下降，固定验证集和固定对手表现下降。
- 非平稳训练：新自博弈分布、replay 窗口或 teacher 比例变化导致目标漂移。
- 优化不匹配：大模型仍使用与小模型相同的学习率、更新次数和样本预算。
- BatchNorm 漂移：非独立同分布样本和阶段性数据混合改变运行统计量。

## 3. 外部接口约束

新架构内部预测 25 个柱子，但对现有调用者继续提供：

```text
input:  float tensor [N, 2, 6, 5, 5]
output: log_policy [N, 150], value [N, 1]
```

模型配置新增架构元数据：

```json
{
  "architecture": "gravity_resnet_v1",
  "policy_space": "columns25",
  "backbone_type": "layer2d",
  "num_channels": 128,
  "num_res_blocks": 6,
  "global_context_blocks": 1,
  "normalization": "group_norm"
}
```

加载器按 `architecture` 分派：

- `modern`：当前 `Connect4Net`。
- `legacy-v21`：现有旧版网络。
- `gravity_resnet_v1`：本设计的新网络。

旧 checkpoint 不迁移；新 checkpoint 继续使用现有 v2 自包含格式。

## 4. 25 列策略空间

### 4.1 动作定义

150 维动作仍按 `(layer, row, col)` 编码。由于重力约束，每个 `(row, col)` 当前最多只有一个合法 `layer`，所以网络只需要输出：

```text
column_logits [N, 5, 5]
```

对每个柱子计算第一个合法空层：

```text
legal_layer[row, col] = first supported empty layer
```

已满柱子在 25 维 softmax 前被屏蔽。合法列概率随后 scatter 到对应的 150 维 action；其他 action 输出有限的大负数或零概率。

### 4.2 训练目标兼容

现有 150 维 MCTS policy target 可折叠为：

```text
target_column[row, col] = sum_layer target_policy[layer, row, col]
```

正常样本中每个柱子至多一个合法层非零。第一版可保留 150 维 loss 接口，由模型内部展开；后续可直接使用 25 维 masked cross entropy，减少无意义计算。

必须处理的边界情况：

- 满棋盘不应进入常规网络搜索；模型仍需返回有限且可归一化的安全结果。
- 禁止用 `0 × -inf` 计算交叉熵，训练路径使用 masked logits 或有限负数。
- 任何展开后的非零概率都必须对应当前合法 action。

### 4.3 策略头

主干输出 `[N, C, 5, 5]` 后，使用共享 1×1 Conv2d 直接得到每个柱子的 logit：

```text
feature [N,C,5,5]
  → GroupNorm/activation
  → Conv2d(C,1,kernel=1)
  → masked log_softmax over 25 columns
```

该结构同时减少全连接参数，并保持平面平移与 D4 对称变换的一致性。

## 5. 重力感知输入

推荐主方案将高度层展开到通道维：

```text
current pieces:  6 channels
opponent pieces: 6 channels
normalized column height: 1 channel
legal/non-full column mask: 1 channel
total: 14 channels over a 5×5 plane
```

其中高度和合法性完全由当前棋盘确定，不引入未来信息。六个高度通道具有固定的绝对高度含义，符合重力游戏中上下方向不对称的事实。

数据增强继续采用 5×5 平面的 D4 旋转和镜像；高度通道不随平面旋转发生层次交换。

保留一个对照方案：

- 输入仍为 `[N,2,6,5,5]`。
- 将每个 3×3×3 卷积分解为 `(1,3,3)` 水平卷积和 `(3,1,1)` 垂直卷积。
- 用于判断 2D 层通道方案是否损失了有用的垂直平移共享能力。

## 6. 主干结构

### 6.1 Gravity Pre-activation Residual Block

默认 block：

```text
x
 ├──────────────────────────────┐
 └→ GroupNorm → SiLU → Conv3×3 ─┤
                  → GroupNorm   │
                  → SiLU        │
                  → Conv3×3 ────┘ + x
```

设计约定：

- 使用 pre-activation，便于较深模型优化。
- GroupNorm 的 group 数自动选择为通道数的因数。
- 第二个卷积采用零初始化或很小的 residual scale，使初始网络接近恒等映射。
- 第一轮实验同时保留 ReLU 对照，SiLU 只有在吞吐损失可接受时才启用。
- mini 可使用深度可分离或 bottleneck block，但必须以实际 CPU/GPU 延迟决定，不仅比较参数量。

### 6.2 全局上下文模块

5×5 平面只有 25 个 token，全局注意力成本较低。balanced/flagship 可在卷积主干后加入少量全局模块：

```text
[N,C,5,5] → [N,25,C] → self-attention/MLP → [N,C,5,5]
```

约束：

- mini 默认不使用全局注意力。
- balanced 最多 1 个 global block。
- flagship 初始候选为 2 个 global blocks。
- 第一版不使用破坏 D4 对称性的任意绝对二维位置嵌入；如需位置偏置，使用中心距离或相对几何偏置。

## 7. 值头

替换当前展平全连接值头：

```text
feature [N,C,5,5]
  → 1×1 Conv/Norm/activation
  → concatenate(global mean, global max)
  → small MLP
  → tanh scalar
```

mini 使用较小隐藏层；flagship 可使用 attention pooling。所有档位均输出当前 canonical player 视角的 `[-1,1]` value。

## 8. 可选训练期辅助头

辅助头只用于提高样本效率，导出纯权重时可以删除：

- 当前玩家一步必胜柱子：25 维多标签目标。
- 对手一步必胜、当前必须阻挡的柱子：25 维多标签目标。
- 每个柱子的合法落子高度：分类或回归目标。
- 可选终局剩余步数区间，用于改善 value 的局面阶段感知。

辅助 loss 初始总权重不超过主 policy/value loss 的 10%，并通过消融实验决定是否保留。不得用规则辅助头直接覆盖正式策略输出。

## 9. 三档候选规格

以下是第一轮实验起点，不是上线前固定值：

| 档位 | 主干宽度 | Residual blocks | Global blocks | 初步参数目标 | 使用场景 |
|---|---:|---:|---:|---:|---|
| mini | 64 | 4 | 0 | 约 0.3–0.8M | CPU、移动端、低延迟 |
| balanced | 128 | 6 | 1 | 约 2–4M | 默认训练和对局 |
| flagship | 192 | 8 | 2 | 约 6–10M | 单 GPU、高棋力、较高时间预算 |

如果 2D 主干容量不足，优先依次增加：

1. block 数；
2. 少量全局 block；
3. 通道宽度；

不优先直接恢复到 256 channels × 8 个普通 3D block。

三档共享同样的输入表示和头部语义，使旗舰到 mini 的蒸馏更直接。

## 10. 蒸馏方案

### 10.1 教师目标

mini/balanced 使用旗舰或现有稳定 teacher 产生：

- MCTS 访问次数归一化后的 25 列 policy target。
- teacher value。
- 最终对局 outcome。
- 可选中间特征，不作为第一版必需项。

优先蒸馏搜索后的 policy，而不是仅蒸馏网络原始 policy 或最终落子。

### 10.2 Loss

建议第一版：

```text
L = w_search_policy * CE(search_policy)
  + w_teacher_policy * KL(teacher_policy || student_policy)
  + w_value * MSE(student_value, blended_value_target)
  + w_aux * auxiliary_losses
```

`blended_value_target` 可混合 teacher value 与真实 outcome。所有权重继续通过当前蒸馏阶段表控制，不把模型档位硬编码到 loss 中。

## 11. 响应档位与 MCTS 联合配置

模型档位不能单独代表最终响应速度。正式比较使用固定墙钟时间：

```text
最终棋力 = 单次评估质量 × 时间预算内完成的有效搜索量
```

建议部署 profile：

| Profile | 模型 | 搜索控制 |
|---|---|---|
| instant | mini | 很低时间预算，优先首批响应 |
| mini | mini | 中低 MCTS 时间预算 |
| balanced | balanced | 默认时间预算 |
| flagship | flagship 或 balanced | 较高时间预算，动态批推理 |

需要允许 balanced 在更高 MCTS 预算下与 flagship 竞争；如果它在相同响应时间内更强，则旗舰档可以采用 balanced 权重加更深搜索，而不必强制使用最大的网络。

## 12. 实验矩阵

### 12.1 架构消融

先使用相同数据、seed、训练更新数和近似参数预算：

| ID | 主干 | 策略头 | 值头 | 目的 |
|---|---|---|---|---|
| A0 | 当前 Conv3d ResNet | 150 FC | flatten FC | 基线 |
| A1 | 当前 Conv3d ResNet | 25 column | flatten FC | 单测动作空间收益 |
| A2 | 当前 Conv3d ResNet | 25 column | pooled value | 单测头部收益 |
| A3 | layer-as-channel Conv2d | 25 column | pooled value | 主候选 |
| A4 | factorized Conv3d | 25 column | pooled value | 垂直共享对照 |
| A5 | A3 + pre-activation GroupNorm | 25 column | pooled value | 稳定性对照 |
| A6 | A5 + global context | 25 column | pooled value | balanced/flagship 增益 |

只在 A3/A4/A5 中选出稳定主干后，再扩大三档规格。

### 12.2 性能指标

- 参数量和纯权重文件大小。
- CPU batch=1 推理 p50/p95。
- GPU batch=1、8、32、64 的推理吞吐和显存。
- MCTS inference requests/s、nodes/s 和 batch 填充率。
- 每手固定 MCTS 次数延迟。
- 每手固定 100 ms、500 ms、3 s 时间预算下完成的模拟数。
- 完整自博弈 games/hour。

### 12.3 学习与稳定性指标

- 固定验证局面上的 policy CE/KL、top-1/top-3。
- value MSE、符号准确率和校准误差。
- 一步必胜、必须阻挡、双威胁等固定 tactical suite。
- 三个 seed 的中位数及方差。
- 最近多个 checkpoint 对固定 teacher/best 的胜率曲线。
- 训练 loss 下降而固定评估退化时，才标记为过拟合。

### 12.4 棋力验收

- 每组至少 200 局并交替先后手；最终候选建议 400 局。
- 同时比较固定 MCTS 次数和固定墙钟时间。
- 对当前 fast、balanced、large 以及固定 teacher 建立完整交叉对局矩阵。
- policy 必须有限、归一化，且非法 action 概率为零。
- 三次独立训练中不得出现持续 value 发散或胜率单调坍塌。

## 13. 实施阶段

### 阶段 M1：兼容头部

- 新增 25 列 policy head 和 150 action scatter 映射。
- 新增 pooled value head。
- 保持当前 Conv3d 主干不变。
- 完成 target collapse、合法性和对称性测试。

### 阶段 M2：重力感知主干

- 实现 `layer2d` 和 `factorized3d` 两个候选。
- 引入 pre-activation GroupNorm block。
- 运行 A0–A5 小规模训练和吞吐消融。

### 阶段 M3：分级模型

- 加入可选 global context block。
- 固化 mini/balanced/flagship 三档配置。
- 完成旗舰到 balanced/mini 的蒸馏。

### 阶段 M4：部署与默认值

- 为 arena、human eval、蒸馏和导出工具加入新架构发现。
- 根据目标机器实测确定时间预算 profile。
- 只有达到性能、稳定性和棋力门槛后，才修改默认模型档位。

## 14. 测试要求

- 25↔150 policy 映射往返及满柱测试。
- 任何非终局都恰好最多有 25 个候选 action，非零概率只位于合法 action。
- D4 旋转/镜像前后的 policy 严格对应。
- 新旧架构 checkpoint 自动识别和加载。
- teacher/student 架构不同仍可蒸馏。
- mini/balanced/flagship 均能完成 CPU 保存→加载→推理。
- AMP FP16/BF16 输出有限，FP32 为默认基线。
- 导出模型不包含仅训练使用的辅助头时，正式 policy/value 输出保持一致。

## 15. 上机前需要确定的信息

- GPU 型号、显存和是否支持 BF16。
- CPU 物理核/逻辑核数和内存。
- 目标部署是否包含纯 CPU、移动 CPU 或仅 NVIDIA GPU。
- 用户可接受的 instant、mini、balanced、flagship 单步响应时间。
- 可用于架构消融的训练时长和最多并行实验数。

拿到这些信息后，先做短训练筛选 A1–A5，再把计算资源集中到最有希望的主干和三档规模，不直接同时完整训练所有组合。
