import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from hgnn import HGNN
from mlp import MLP
from env import USVSchedulingEnv

class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, hidden_dim=8, hgnn_layers=3):
        super().__init__()
        self.hgnn = HGNN(hidden_dim=hidden_dim, n_layers=hgnn_layers, n_neighbors=2)

        # Actor网络：输入 [task, usv, global] -> 输出 score
        # task(1) + usv(1) + global(2) = 4 * hidden_dim
        self.actor = MLP(
            input_dim=hidden_dim * 4,  # <--- 修改这里，由 3 改为 4
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

        # Critic网络：输入 [global] -> 输出 value
        # global(2) = 2 * hidden_dim
        self.critic = MLP(
            input_dim=hidden_dim * 2,  # 这里是对的
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

    def forward(self, state_dict, available_actions_list):
        """
        前向传播 (支持 Batch)

        Args:
            state_dict: 字典，每个值为 Batch Tensor
                - usv_features: [batch, n_usvs, 4]
                - task_features: [batch, n_tasks, 5]
            available_actions_list: list of list of tuples, 长度为 batch_size
                                    每个元素是该样本的可用动作列表 [(t, u), ...]
        """
        # 1. 获取图嵌入
        embeddings = self.hgnn(state_dict)
        usv_embed = embeddings['usv_embed']  # [batch, n_usvs, hidden]
        task_embed = embeddings['task_embed']  # [batch, n_tasks, hidden]
        global_embed = embeddings['global_embed']  # [batch, 2*hidden]

        batch_size = usv_embed.size(0)

        # 2. 计算 Critic Value
        values = self.critic(global_embed)  # [batch, 1]

        # 3. 计算 Actor Logits (处理动态动作空间)
        # 由于每个样本的动作数量不同，无法直接输出一个 Tensor 矩阵
        # 我们需要返回一个列表，其中包含每个样本对应动作的 logits

        batch_logits = []

        for i in range(batch_size):
            actions = available_actions_list[i]  # 当前样本的动作列表 [(t, u), ...]
            if not actions:
                batch_logits.append(torch.tensor([], device=usv_embed.device))
                continue

            # 提取对应的嵌入
            # actions 是 list of tuples
            task_ids = [a[0] for a in actions]
            usv_ids = [a[1] for a in actions]

            # task_embed[i]: [n_tasks, hidden] -> 选出 [n_acts, hidden]
            curr_task_emb = task_embed[i, task_ids, :]
            curr_usv_emb = usv_embed[i, usv_ids, :]

            # global_embed[i]: [2*hidden] -> 扩充为 [n_acts, 2*hidden]
            curr_global = global_embed[i].unsqueeze(0).expand(len(actions), -1)

            # 拼接: [n_acts, 4*hidden]
            actor_input = torch.cat([curr_task_emb, curr_usv_emb, curr_global], dim=-1)

            # 计算分数: [n_acts, 1] -> [n_acts]
            scores = self.actor(actor_input).squeeze(-1)
            batch_logits.append(scores)

        return batch_logits, values

    def get_action(self, state_dict, available_actions, deterministic=False):
        """
        单步推理选择动作
        state_dict: 单个样本的数据 (已 unsqueeze 为 batch=1)
        available_actions: 单个样本的动作列表
        """
        batch_logits, value = self.forward(state_dict, [available_actions])
        logits = batch_logits[0]  # [n_actions]

        if logits.numel() == 0:
            return None, None, value

        # Softmax 得到概率
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action_idx = torch.argmax(probs).item()
        else:
            dist = Categorical(probs)
            action_idx = dist.sample().item()

        log_prob = torch.log(probs[action_idx] + 1e-8)

        return action_idx, log_prob, value


class PPO:
    def __init__(self, hidden_dim=8, hgnn_layers=3, lr=3e-4,
                 gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        # PPO 不需要 target network, 但需要 old policy 用于计算 ratio
        # 这里实际上可以只存参数，或者在一个网络里做，通常不需要完整的 policy_old 对象，
        # 只需要在 update 开始时 detach 出来的旧 log_probs。
        # 但为了标准 PPO 实现 (使用 old_policy 前向)，保留它。
        self.policy_old = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Buffer: list of (state, avail_actions, action_idx, reward, log_prob, done, value)
        self.buffer = []

    def select_action(self, env, state_dict, deterministic=False):
        available_actions = env.get_available_actions()  # 返回 ID 列表

        # 将 ID 转为 (task, usv) 对
        action_pairs = [(a // env.n_usvs, a % env.n_usvs) for a in available_actions]

        state_tensor = self._prepare_state(state_dict)  # batch=1

        # 使用 policy_old 进行采样 (PPO标准)
        action_idx_in_list, log_prob, value = self.policy_old.get_action(
            state_tensor, action_pairs, deterministic
        )

        if action_idx_in_list is None:
            # 修正：这里原来只有4个None，必须返回5个None以匹配 main.py 的解包
            return None, None, None, None, None

        # 选中的动作 ID
        real_action_id = available_actions[action_idx_in_list]

        # 返回: 真实动作ID, 动作在列表中的索引, log_prob, value, 可用动作对(用于存buffer)
        return real_action_id, action_idx_in_list, log_prob, value, action_pairs

    def _prepare_state(self, state_dict, batch=False):
        """
        转换状态为 Tensor。支持混合规模输入的自动 Padding。
        """
        if not batch:
            # 单个样本处理 (保持不变)
            return {
                'usv_features': torch.FloatTensor(state_dict['usv_features']).unsqueeze(0).to(self.device),
                'task_features': torch.FloatTensor(state_dict['task_features']).unsqueeze(0).to(self.device)
            }
        else:
            # === 修复：混合规模 Batch 处理 (Padding) ===
            # state_dict 是列表: [{'usv_features': ..., 'task_features': ...}, ...]

            # 1. 获取当前 Batch 中最大的维度
            n_samples = len(state_dict)
            max_n_usvs = max(s['usv_features'].shape[0] for s in state_dict)
            max_n_tasks = max(s['task_features'].shape[0] for s in state_dict)

            # 获取特征维度 (假设所有样本特征维度一致)
            usv_dim = state_dict[0]['usv_features'].shape[1]  # 通常是 4
            task_dim = state_dict[0]['task_features'].shape[1]  # 通常是 5

            # 2. 初始化全零 Padding Tensor
            padded_usvs = torch.zeros((n_samples, max_n_usvs, usv_dim), dtype=torch.float32).to(self.device)
            padded_tasks = torch.zeros((n_samples, max_n_tasks, task_dim), dtype=torch.float32).to(self.device)

            # 3. 填充数据并处理 Mask
            for i, s in enumerate(state_dict):
                u_data = s['usv_features']  # numpy array
                t_data = s['task_features']  # numpy array

                curr_usvs = u_data.shape[0]
                curr_tasks = t_data.shape[0]

                # 复制真实数据
                padded_usvs[i, :curr_usvs, :] = torch.from_numpy(u_data).to(self.device)
                padded_tasks[i, :curr_tasks, :] = torch.from_numpy(t_data).to(self.device)

                # --- 关键 Mask 处理 ---
                # 对于填充的任务，将 Scheduled (索引4) 设为 1
                # 这样 hgnn.py 中的 adj_matrix 计算会将其 Mask 掉 (1 - 1 = 0)
                if curr_tasks < max_n_tasks:
                    padded_tasks[i, curr_tasks:, 4] = 1.0
                    # 可选：将填充任务坐标设为无穷远，避免干扰 KNN
                    padded_tasks[i, curr_tasks:, 0:2] = 99999.0

            return {
                'usv_features': padded_usvs,
                'task_features': padded_tasks
            }

    def store_transition(self, state, avail_actions, action_idx, reward, log_prob, done, value):
        """存储单步数据"""
        self.buffer.append({
            'state': state,
            'avail_actions': avail_actions,  # list of tuples
            'action_idx': action_idx,  # int
            'reward': reward,
            'log_prob': log_prob.item(),  # float
            'done': done,
            'value': value.item()  # float
        })

    def update(self, n_epochs=4, batch_size=32):
        if len(self.buffer) == 0: return {}

        # 1. 计算 GAE / Returns
        rewards = [t['reward'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]
        values = [t['value'] for t in self.buffer]

        returns = []
        advantages = []
        gae = 0

        # 假设最后一个状态之后 value 为 0 (或者需要在外部传入 next_value)
        # 简单起见，这里假设 trajectory 结束
        next_value = 0

        for i in reversed(range(len(self.buffer))):
            mask = 1 - dones[i]
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * 0.95 * mask * gae  # lambda=0.95

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

            next_value = values[i]

        # 转换为 Tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # 归一化 Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备 Batch 数据
        states_list = [t['state'] for t in self.buffer]
        avail_actions_list = [t['avail_actions'] for t in self.buffer]
        old_action_idxs = torch.tensor([t['action_idx'] for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.buffer], dtype=torch.float32, device=self.device)

        # 2. PPO Update Epochs
        dataset_size = len(self.buffer)
        indices = np.arange(dataset_size)

        total_loss = 0

        for _ in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                # 构造 Mini-batch
                mb_states_list = [states_list[i] for i in mb_idx]
                mb_avail_actions = [avail_actions_list[i] for i in mb_idx]
                mb_action_idxs = old_action_idxs[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                # 转换 State 到 Tensor
                mb_state_tensor = self._prepare_state(mb_states_list, batch=True)

                # 前向传播 - 获取当前策略下的 Logits 和 Value
                # 注意：这里需要 forward 支持 batch 处理，但每个样本 action 数量不同
                batch_logits, batch_values = self.policy.forward(mb_state_tensor, mb_avail_actions)

                # 计算 Log Probs 和 Entropy
                curr_log_probs = []
                curr_entropy = []

                for i, logits in enumerate(batch_logits):
                    if logits.numel() == 0:
                        # 异常处理：无动作
                        curr_log_probs.append(torch.tensor(0.0, device=self.device))
                        curr_entropy.append(torch.tensor(0.0, device=self.device))
                        continue

                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)

                    # 获取当前 batch 中该样本实际执行动作的新 log_prob
                    act_idx = mb_action_idxs[i]
                    curr_log_probs.append(dist.log_prob(act_idx))
                    curr_entropy.append(dist.entropy())

                curr_log_probs = torch.stack(curr_log_probs)
                curr_entropy = torch.stack(curr_entropy).mean()
                batch_values = batch_values.squeeze(-1)  # [batch]

                # PPO Loss
                ratio = torch.exp(curr_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(batch_values, mb_returns)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * curr_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        # Update Old Policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {'loss': total_loss / n_epochs}

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


# ============================================================================
# Train Loop 修正
# ============================================================================
def train_hgnn_ppo(n_epochs=1000, test_every=10):
    config = InstanceConfig(n_usvs=4, n_tasks=40)
    generator = InstanceGenerator(config)
    ppo = PPO(hidden_dim=16, hgnn_layers=2)  # 调整超参

    train_rewards = []

    for epoch in range(n_epochs):
        instance = generator.generate(seed=epoch)
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        ep_reward = 0

        # 收集轨迹
        while not done:
            # 获取动作
            real_action, action_idx, log_prob, value, avail_acts = ppo.select_action(env, state)

            if real_action is None: break

            next_state, reward, done, _ = env.step(real_action)

            # 存入 Buffer
            # 注意存的是 action_idx (列表中的索引)，方便后面从 avail_acts 中恢复 logit
            ppo.store_transition(state, avail_acts, action_idx, reward, log_prob, done, value)

            ep_reward += reward
            state = next_state

        train_rewards.append(ep_reward)

        # 更新 (每个 Episode 更新一次，或者积攒多个)
        ppo.update()

        if (epoch + 1) % test_every == 0:
            print(f"Epoch {epoch + 1} | Reward: {np.mean(train_rewards[-test_every:]):.2f}")

    return ppo, train_rewards