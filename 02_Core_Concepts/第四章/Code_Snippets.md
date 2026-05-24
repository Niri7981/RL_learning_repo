## 4.2

### 定义 Q 网络

```python
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 输入层 -> 隐藏层
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 隐藏层 -> 输出层
        )

    def forward(self, x):
        return self.net(x)  # 输出形状: (batch_size, action_dim)
```

### 定义经验回放

```python
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),      # (B, state_dim)
            torch.LongTensor(actions),      # (B,)
            torch.FloatTensor(rewards),     # (B,)
            torch.FloatTensor(next_states), # (B, state_dim)
            torch.FloatTensor(dones),       # (B,)
        )

    def __len__(self):
        return len(self.buffer)
```

- **`capacity=10000`**：CartPole 500 局训练会产生数千到数万条转移，`10000` 条大约保留最近几十到一百局。太小经验不够多样，太大保留过时经验会拖慢收敛。LunarLander 常用规模是 $10^4$ 到 $10^5$，Atari 常用规模是 $10^5$ 到 $10^6$。
- **`deque(maxlen=capacity)`**：满了自动淘汰最旧元素；头删是 $O(1)$，比 `list` 更适合做队列。
- **`FloatTensor` 和 `LongTensor`**：状态、奖励等数值参与运算，用 `FloatTensor`；动作用 `LongTensor`，因为 `gather` 需要整数索引；`dones` 用 `FloatTensor`，因为后面要参与 $(1-d)$ 的乘法。

### 定义损失函数与参数更新

```python
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma

        # Q 网络（学生）和目标网络（阅卷老师）
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)
```

这段代码就是整个 DQN 算法的控制中枢，也就是 `DQNAgent`。它把前面零散讨论的几个组件重新装回一个类里：

- 函数逼近器，也就是学生网络
- 目标网络，也就是阅卷老师
- 优化器 `Adam`
- 经验回放池 `ReplayBuffer`

## 双网络物理隔离与参数克隆

这里的核心目的，是解决“动了当前 $Q$，下一个状态的 $\max Q$ 也跟着乱变”的训练崩溃问题。

```python
# Q 网络（学生）和目标网络（阅卷老师）
self.q_net = QNetwork(state_dim, action_dim)
self.target_net = QNetwork(state_dim, action_dim)
```

- **内存状态**：这一步在内存中创建了两套完全独立的网络结构。它们一开始都是随机初始化，所以参数默认不同。

```python
self.target_net.load_state_dict(self.q_net.state_dict())
```

- **做什么**：把学生网络的全部参数强行拷贝给目标网络。
- **目的**：让学生和老师从同一个起点出发，避免训练一开始目标值和预测值就处在完全不同的量级。

```python
self.target_net.eval()
```

- **做什么**：把目标网络切换到评估模式。
- **目的**：关闭 Dropout、BatchNorm 这类训练态组件，再配合后面的 `torch.no_grad()`，把目标网络固定成一个只负责出“标准答案”的静态副本。

## 驱动与存储组件挂载

```python
self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
```

- **做什么**：实例化一个 `Adam` 优化器。
- **关键点**：这里传进去的只有 `self.q_net.parameters()`，也就是说只有学生网络的参数会被梯度下降更新，目标网络不会被优化器直接改动。

```python
self.buffer = ReplayBuffer(capacity=10000)
```

- **做什么**：创建经验回放池。
- **作用**：把每一步产生的五元组 $(s,a,r,s',d)$ 记录下来，供后续随机采样训练使用。

## PyTorch 的本质：一个带自动求导的矩阵计算库

在没有深度学习框架时，人们常用 `NumPy` 做矩阵运算，但它有两个典型痛点：

1. 大规模计算默认只能用 CPU，面对海量矩阵乘法时会非常慢。
2. 梯度得自己手推，更新参数时手写微积分公式会非常痛苦。

PyTorch 解决的两件大事是：

- **第一**：把矩阵运算搬到 GPU 上并行执行。
- **第二**：提供自动求导（*autograd*），你只写前向传播，反向传播需要的梯度它会自动帮你算。

### 为什么 `target_net` 要同步初始化？

如果目标网络和 Q 网络从两套完全无关的随机参数开始，两个网络的 $Q$ 值输出可能一开始就在不同量级，比如一个输出 $-5.0$，另一个输出 $200.0$。这会直接把 TD Target 拉爆，导致 MSE loss 爆炸，训练早期就把参数推飞。

所以这句：

```python
self.target_net.load_state_dict(self.q_net.state_dict())
```

就是为了保证两个网络至少从同一个量级起步。

### 核心的 `update()` 方法

```python
def update(self, batch_size):
    """核心更新：一个 batch 的前向传播 + 反向传播"""
    if len(self.buffer) < batch_size:
        return 0.0

    # 从回放池采样一个 batch
    states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

    # Q 网络前向传播
    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 目标网络前向传播
    with torch.no_grad():
        next_q_max = self.target_net(next_states).max(dim=1)[0]
        targets = rewards + self.gamma * next_q_max * (1 - dones)

    # 计算 MSE Loss
    loss = nn.MSELoss()(q_values, targets)

    # 反向传播与参数更新
    self.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(self.q_net.parameters(), max_norm=10)
    self.optimizer.step()

    return loss.item()
```

## 核心操作：学生预测与点对点捞取 `gather`

```python
q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
```

这一行是 DQN 里最关键的张量操作之一。

它的含义是：

1. `self.q_net(states)` 先输出一个形状为 `(B, action_dim)` 的矩阵，表示 batch 中每个状态下所有动作的 $Q$ 值。
2. `actions.unsqueeze(1)` 把动作索引从一维拉成列向量。
3. `gather(1, ...)` 沿着动作维，把“当前样本实际执行的那个动作”对应的 $Q$ 值精准捞出来。
4. `.squeeze(1)` 再把多余维度压掉。

最后得到的 `q_values`，就是一维数组，里面每个元素都对应“这个样本里实际执行动作的预测 $Q$ 值”。

## 标准答案生成：目标网络的物理隔离

```python
with torch.no_grad():
    next_q_max = self.target_net(next_states).max(dim=1)[0]
    targets = rewards + self.gamma * next_q_max * (1 - dones)
```

这里的三层保护分别是：

- **`torch.no_grad()`**：明确告诉 PyTorch，这一段只算数值，不记录梯度。
- **`self.target_net(next_states)`**：用被冻结的目标网络来算下一状态价值，而不是用正在不断变化的学生网络。
- **`(1 - dones)`**：如果游戏终止，未来价值直接清零。

## 误差清算与参数更新

```python
loss = nn.MSELoss()(q_values, targets)
```

- **含义**：均方误差（MSE），用来衡量学生预测和标准答案之间的平均平方距离。

```python
self.optimizer.zero_grad()
loss.backward()
clip_grad_norm_(self.q_net.parameters(), max_norm=10)
self.optimizer.step()
```

这四步是 PyTorch 里标准的更新流程：

1. `zero_grad()`：清掉上一轮残留梯度
2. `backward()`：反向传播，计算当前 loss 对所有参数的梯度
3. `clip_grad_norm_()`：梯度裁剪，防止梯度爆炸
4. `step()`：让优化器真正更新参数

### 为什么要梯度裁剪？

深度强化学习中的数据分布非常不稳定，奖励和目标值可能会突然剧烈波动。如果梯度过大，就可能直接把参数推成 `NaN`。梯度裁剪相当于在工程上加了一道保险丝：再大的梯度，也只能在规定范围内更新。

## `update()` 过程的简明总结

- **缓冲区检查**：样本不够一个 batch 就先不更新
- **`gather`**：从所有动作的输出里，挑出实际动作对应的 $Q$ 值
- **`torch.no_grad()`**：冻结目标网络，不让梯度流进去
- **`.max(dim=1)[0]`**：取下一状态的最大动作价值
- **TD Target**：`rewards + gamma * next_q_max * (1 - dones)`
- **MSE Loss**：对大误差的惩罚比 L1 更强
- **固定四步**：`zero_grad -> backward -> clip -> step`

### 动作选择与目标网络同步

```python
def select_action(self, state, epsilon):
    """epsilon-greedy 动作选择"""
    if random.random() < epsilon:
        return random.randint(0, self.action_dim - 1)
    with torch.no_grad():
        q_values = self.q_net(torch.FloatTensor(state).unsqueeze(0))
    return q_values.argmax(dim=1).item()


def update_target(self):
    """硬更新：将 Q 网络参数复制到目标网络"""
    self.target_net.load_state_dict(self.q_net.state_dict())
```

- **$\epsilon$-greedy**：训练早期预测很差，如果只选 `argmax`，很容易陷入过早利用；所以用概率 $\epsilon$ 强制探索。
- **`unsqueeze(0)`**：把形状从 `(state_dim,)` 扩成 `(1, state_dim)`，补上 batch 维。
- **`argmax(dim=1).item()`**：把网络输出的最佳动作索引转成 Python 整数。
- **硬更新**：直接执行 $\theta^- \leftarrow \theta$，这是 DQN 论文里的经典做法。

### 训练循环

```python
import gymnasium as gym


num_episodes = 500
batch_size = 64
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.01, 0.995
target_update_freq = 10

env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2)
epsilon = epsilon_start

for episode in range(num_episodes):
    state, _ = env.reset()
    while True:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.update(batch_size)
        state = next_state
        if done or truncated:
            break

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    if (episode + 1) % target_update_freq == 0:
        agent.update_target()
```

整个实现的核心，始终还是 `update()`：前向传播算预测，目标网络算 TD Target，MSE 算 loss，反向传播更新参数。其余部分，本质上都是在为这件事服务。
