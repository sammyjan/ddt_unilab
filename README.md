概述：本仓库在UniLab基础上加入本末科技D1系列机器人训练任务。UniLab支持多平台配置，如Linux CUDA, Linux ROCm, Linux XPU, and Apple Silicon / macOS，但本仓库目前只在Linux CUDA进行过验证，其他平台未知是否可用，感兴趣的开发者可自行尝试。

## 关于安装
```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/sammyjan/ddt_unilab.git
cd UniLab
make setup
```
更多细节请参考UniLab官方https://github.com/unilabsim/UniLab 以及
https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/2-installation.html

## 1. 快速开始

### 最小训练命令：以D1为例

```bash
uv run train --algo ppo --task d1_flat --sim motrix
```

- `--algo ppo`：选择算法（`ppo`、`appo`、`sac`、`td3`、`flashsac`、`mlx_ppo`）目前仅注册了PPO任务，用别的算法请注册新的任务
- `--task d1_flat`：选择任务名
- `--sim mujoco`：选择仿真后端（`mujoco` 或 `motrix`）
- CLI 会自动拼接为 `task=d1_flat/mujoco` 并从 `conf/ppo/task/d1_flat/mujoco.yaml` 加载配置
- 指定GPU，以英伟达显卡为例：
`CUDA_VISIBLE_DEVICES=0 uv run train --algo ppo --task d1_flat --sim motrix`，其中CUDA_VISIBLE_DEVICES=0：用第 0 张卡
- 训练时没有渲染画面，训练结束后会自动play，此时有渲染画面


---

## 2. 任务与后端选择

训练入口：`uv run train`

格式：`uv run train --algo <算法> --task <任务> --sim <后端>`

| 任务 | 后端配置示例 |
|------|-------------|
| D1 四轮（quadruped wheeled）平地 | `task=d1_flat/mujoco` |
| D1 四轮（quadruped wheeled）平地 | `task=d1_flat/motrix` |
| D1H 双轮足（bipedal wheeled）平地 | `task=d1h_flat/mujoco` |
| D1H 双轮足（bipedal wheeled）平地 | `task=d1h_flat/motrix` |


> 提示：若未指定 `task`，默认使用 `conf/ppo/config.yaml` 中的 `defaults`（通常为 `go1_joystick_flat/mujoco`）。

---

## 3. 常用命令行参数

所有参数均通过 Hydra 覆盖，格式为 `key=value`。

### 3.1 训练规模与迭代

| 参数 | 说明 | 示例 |
|------|------|------|
| `algo.num_envs` | 并行环境数 | `algo.num_envs=4096` |
| `algo.max_iterations` | 最大训练迭代次数 | `algo.max_iterations=10000` |
| `training.num_timesteps` | 按总步数自动计算迭代数（优先级高于 max_iterations） | `training.num_timesteps=100000000` |
| `algo.num_steps_per_env` | 每轮每个环境采集步数 | `algo.num_steps_per_env=24` |

### 3.2 算法超参

| 参数 | 说明 | 示例 |
|------|------|------|
| `algo.seed` | 随机种子 | `algo.seed=42` |
| `algo.policy.init_noise_std` | 策略初始噪声标准差 | `algo.policy.init_noise_std=0.5` |
| `algo.algorithm.learning_rate` | 学习率 | `algo.algorithm.learning_rate=1e-3` |
| `algo.algorithm.entropy_coef` | 熵系数 | `algo.algorithm.entropy_coef=0.01` |
| `algo.algorithm.gamma` | 折扣因子 | `algo.algorithm.gamma=0.99` |
| `algo.algorithm.lam` | GAE lambda | `algo.algorithm.lam=0.95` |
| `algo.algorithm.num_learning_epochs` | 每次数据复用训练轮数 | `algo.algorithm.num_learning_epochs=5` |
| `algo.algorithm.num_mini_batches` | PPO mini-batch 数量 | `algo.algorithm.num_mini_batches=4` |

### 3.3 环境与奖励

| 参数 | 说明 | 示例 |
|------|------|------|
| `env.commands.vel_limit` | 指令速度范围 `[min_x, min_y, min_yaw]` / `[max_x, max_y, max_yaw]` | 数组格式，建议在 YAML 中修改 |
| `env.control_config.Kp` | PD 控制比例增益 | `env.control_config.Kp=40.0` |
| `env.control_config.Kd` | PD 控制微分增益 | `env.control_config.Kd=1.0` |
| `env.control_config.action_scale` | 动作缩放系数 | `env.control_config.action_scale=0.25` |
| `reward.scales.tracking_lin_vel` | 奖励权重：线速度跟踪 | `reward.scales.tracking_lin_vel=2.0` |
| `reward.scales.base_height` | 奖励权重：高度惩罚 | `reward.scales.base_height=-1.0` |

> 所有 `reward.scales.*` 都可在命令行直接覆盖。

### 3.4 日志与恢复

| 参数 | 说明 | 示例 |
|------|------|------|
| `algo.save_interval` | 模型保存间隔（按迭代数） | `algo.save_interval=100` |
| `algo.resume` | 是否从最新 checkpoint 恢复 | `algo.resume=true` |
| `algo.load_run` | 指定恢复的运行目录名 | `algo.load_run=2026-06-25_21-32-20_mujoco` |
| `algo.checkpoint` | 指定恢复的模型编号 | `algo.checkpoint=100` |
| `training.experiment_name` | 实验名称（用于日志目录） | `training.experiment_name=d1_exp` |

---

## 4. 典型训练命令示例

### 4.1 D1Flat 完整训练（MuJoCo 后端）

```bash
uv run train --algo ppo --task d1_flat --sim mujoco \
  algo.num_envs=4096 \
  algo.max_iterations=40000 \
  algo.seed=1 \
  algo.save_interval=500
```

### 4.2 D1HFlat 完整训练（MuJoCo 后端）

```bash
uv run train --algo ppo --task d1h_flat --sim mujoco \
  algo.num_envs=4096 \
  algo.max_iterations=40000 \
  algo.seed=1
```

### 4.3 快速 Smoke Test（调试用）

```bash
uv run train --algo ppo --task d1_flat --sim mujoco \
  algo.max_iterations=10 \
  algo.num_envs=4
```

### 4.4 切换后端（MotrixSim）

```bash
uv run train --algo ppo --task d1_flat --sim motrix
```

### 4.5 修改奖励权重

```bash
uv run train --algo ppo --task d1_flat --sim mujoco \
  reward.scales.tracking_lin_vel=3.0 \
  reward.scales.base_height=-2.0
```

### 4.6 修改控制增益

```bash
uv run train --algo ppo --task d1h_flat --sim mujoco \
  env.control_config.Kp=50.0 \
  env.control_config.Kd=2.0
```

### 4.7 从 Checkpoint 恢复训练

```bash
uv run train --algo ppo --task d1_flat --sim mujoco \
  algo.resume=true \
  algo.load_run=2026-06-25_21-32-20_mujoco \
  algo.checkpoint=500
```

---

## 5. Play / 评估模式

训练完成后，可加载最新模型进行策略回放与视频录制。

### 5.1 自动 Play（训练结束后自动执行）

默认行为：训练结束后会自动加载最新模型并渲染一段视频。

### 5.2 仅 Play（不训练）

```bash
uv run eval --algo ppo --task d1_flat --sim mujoco --load-run -1
```

- `--load-run -1`：加载最新一次训练的 checkpoint
- 如需指定具体运行目录：`--load-run=2026-06-25_21-32-20_mujoco`

---

## 6. 日志与输出目录

训练日志和模型默认保存在：

```
logs/rsl_rl_ppo/<task_name>/<timestamp>_<backend>/
```

目录内容：

| 文件 | 说明 |
|------|------|
| `model_<iter>.pt` | 策略 checkpoint |
| `events.out.tfevents.*` | TensorBoard 日志 |
| `config.yaml` / `config_tree.txt` | 完整配置记录 |
| `play_video.mp4` | 训练结束后的策略回放视频（如渲染成功） |

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl_ppo
```

---

## 7. 配置文件结构

```
conf/ppo/
├── config.yaml              # 顶层默认配置
├── algo/                    # 算法相关配置（如不同学习率策略）
└── task/
    ├── d1_flat/
    │   ├── mujoco.yaml      # D1 + MuJoCo 后端
    │   └── motrix.yaml      # D1 + MotrixSim 后端
    ├── d1h_flat/
    │   ├── mujoco.yaml      # D1H + MuJoCo 后端
    │   └── motrix.yaml      # D1H + MotrixSim 后端
    └── ...                  # 其他机器人任务
```

每个任务 YAML 覆盖以下内容：

- `training.task_name`：环境注册名（如 `D1Flat`、`D1HFlat`）
- `training.sim_backend`：`mujoco` 或 `motrix`
- `algo.*`：网络结构、学习率、噪声等算法参数
- `env.*`：控制参数、指令范围等环境配置
- `reward.scales`：各奖励项权重

---

## 8. 常见问题

### Q: 训练时报 `Environment 'D1Flat' is not registered`

A: 确保 `src/unilab/envs/locomotion/__init__.py` 中已包含 `"unilab.envs.locomotion.d1"`，且 `d1/__init__.py` 正确导入了任务类。

### Q: 如何调整观测噪声？

A: 修改环境类中的 `NoiseConfig` 或 YAML 中对应的 `env.noise_config.*` 字段。

### Q: 视频渲染报错 `unexpected keyword argument 'fps'`

A: 这是 `imageio` 版本兼容问题，不影响模型训练。可忽略或通过安装兼容版本解决：

```bash
pip install imageio[ffmpeg]
```

---

## 9. 附录：D1 / D1H 观测维度速查

| 机器人 | Actor 输入维度 | Critic 输入维度 | 动作维度 |
|--------|---------------|----------------|----------|
| D1     | 57            | 60             | 16       |
| D1H    | 33            | 36             | 8        |
