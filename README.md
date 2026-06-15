# QwenPaw + verl GRPO Integration

This repository contains an integration between QwenPaw desktop application and verl reinforcement learning framework, using GRPO (Group Relative Policy Optimization) algorithm.

## 📁 Files

| File | Description |
|------|-------------|
| `qwenpaw_e2e.py` | Complete GRPO training pipeline with explicit Dummy interfaces |
| `qwenpaw_rollout.py` | QwenPaw client library |
| `gsm8k_manual_preprocess.py` | GSM8K data preprocessing |
| `QWENPAW_README.md` | Detailed documentation in Chinese |

## 🚀 Quick Start

### Prerequisites
1. Install Python 3.10+
2. Install verl dependencies
3. Open QwenPaw desktop application
4. Download Qwen2.5-0.5B-Instruct model

### Run Training

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training
python qwenpaw_e2e.py
```

## 🎯 Features

- **GRPO Algorithm**: Group Relative Policy Optimization implementation
- **Group Sampling**: Multiple rollouts per prompt (configurable `grpo_n`)
- **Relative Advantage**: Reward normalization relative to group
- **Clipped Surrogate Loss**: Stable policy updates

## ⚙️ Configuration

In `qwenpaw_e2e.py`:
```python
grpo_n: int = 4        # Number of rollouts per prompt
clip_eps: float = 0.2  # Clip ratio
max_steps: int = 3     # Training steps
```

## 📝 Notes

This is a demonstration implementation using dummy interfaces. For production use, please refer to verl's official documentation and integrate with the real verl core algorithms.

## 🔗 Related

- [verl Framework](https://github.com/verl-project/verl)
- [GRPO Algorithm Documentation](../docs/algo/grpo.md)
