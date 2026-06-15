# QwenPaw + verl GRPO 训练

本目录包含了将 QwenPaw 桌面应用集成到 verl GRPO 训练流程的脚本。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `qwenpaw_e2e.py` | **主脚本！** 完整的 GRPO 训练流程（显式 Dummy 接口） |
| `qwenpaw_rollout.py` | QwenPaw 客户端库 |
| `gsm8k_manual_preprocess.py` | GSM8K 数据预处理 |

## 🚀 快速开始

### 前置条件
1. 安装 Python 3.10+ 虚拟环境
2. 安装 verl 依赖
3. 打开 QwenPaw 桌面应用
4. 下载 Qwen2.5-0.5B-Instruct 模型

### 运行训练

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行训练
python qwenpaw_e2e.py
```

## 📋 训练流程

脚本执行以下步骤：

1. **初始化 Policy Model**：加载 Qwen2.5-0.5B-Instruct 模型
2. **初始化 QwenPaw 客户端**：连接到桌面应用
3. **加载 GSM8K 数据**：从预处理的 parquet 文件读取
4. **执行训练循环**（共 3 步）：
   - Rollout：调用 QwenPaw 生成响应
   - Dummy 处理：生成 token-level log_probs、ref_log_probs、values 等
   - 奖励计算：GSM8K 验证答案正确性
   - GRPO 优势计算：分组归一化（模仿 verl core_algos）
   - 模型更新：真实的权重更新

## 🎯 Dummy 接口说明

由于 QwenPaw 目前只返回纯文本响应，我们用显式 Dummy 函数模拟 verl 期望的数据结构：

| Dummy 函数 | 说明 |
|------------|------|
| `dummy_compute_log_probs_from_text()` | 用 Policy Model 计算 token-level log_probs |
| `dummy_compute_ref_log_probs()` | 模拟 Reference Model 的 log_probs |
| `dummy_critic_forward()` | 模拟 Critic 网络的值估计 |
| `dummy_token_level_rewards()` | 模拟 token-level 奖励 |
| `compute_mock_grpo_outcome_advantage()` | 模仿 verl core_algos 的优势计算 |
| `compute_mock_ppo_clip_loss()` | 模仿 verl PPO Clip Actor 损失 |

## ⚙️ 配置说明

在 `qwenpaw_e2e.py` 顶部可以调整：

```python
QWENPAW_PORT = 52143  # QwenPaw 端口
MODEL_PATH = "models/qwen2.5-0.5b-instruct"  # 模型路径
TOTAL_STEPS = 3  # 训练步数
```

## 📝 关键文件

- `qwenpaw_rollout.py`：包含 `QwenPawClient` 类，处理与桌面应用的 API 通信
- `gsm8k_manual_preprocess.py`：将 GSM8K 转换为 verl 格式

## 💡 下一步

要完全集成到 verl 的核心算法，需要：

1. 修改 QwenPaw 返回 token-level 的 log_probs
2. 集成 Reference Model
3. 集成 Critic Model
4. 使用 verl 的 `PPOTrainer` 或 `GRPOTrainer`
