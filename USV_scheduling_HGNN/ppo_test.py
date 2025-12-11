# ============================================================================
# 5. ppo.py - PPO强化学习算法 (引入自注意力机制版)
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# 假设这些模块在同一目录下
from hgnn import HGNN
from mlp import MLP


class ActorCritic(nn.Module):
    """
    Actor-Critic网络
    优化：引入自注意力机制处理变长动作空间，并更新状态表示
    """

    def __init__(self, hidden_dim=64, hgnn_layers=2, n_heads=4):
        super().__init__()
        self.hgnn = HGNN(hidden_dim=hidden_dim, n_layers=hgnn_layers, n_neighbors=2)

        # 嵌入维度定义
        # Task (hidden) + USV (hidden) = Action Vector (2 * hidden)
        # Global (2 * hidden) = State Vector
        self.embed_dim = hidden_dim * 2

        # --- 新增：自注意力模块 ---
        # 用于让 State 和所有 Available Actions 进行交互
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=n_heads)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Actor网络：输入 [Updated_Action(2h), Updated_State(2h)] -> 输出 score
        self.actor = MLP(
            input_dim=self.embed_dim * 2,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

        # Critic网络：输入 [Original_State(2h)] -> 输出 value
        # Critic 通常基于原始状态评估 V(s)
        self.critic = MLP(
            input_dim=self.embed_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

    def forward(self, state_dict, available_actions_list):
        """
        前向传播 (支持 Batch)
        """
        # 1. 获取图嵌入
        embeddings = self.hgnn(state_dict)
        usv_embed = embeddings['usv_embed']  # [batch, n_usvs, hidden]
        task_embed = embeddings['task_embed']  # [batch, n_tasks, hidden]
        global_embed = embeddings['global_embed']  # [batch, 2*hidden] (State Vector)

        batch_size = usv_embed.size(0)

        # 2. 计算 Critic Value (基于全局状态)
        values = self.critic(global_embed)  # [batch, 1]

        # 3. 计算 Actor Logits (引入自注意力更新)
        batch_logits = []

        for i in range(batch_size):
            actions = available_actions_list[i]  # 当前样本的动作列表 [(t, u), ...]

            if not actions:
                batch_logits.append(torch.tensor([], device=usv_embed.device))
                continue

            # --- 构建序列输入 ---
            # A. 提取动作向量
            task_ids = [a[0] for a in actions]
            usv_ids = [a[1] for a in actions]

            # [n_acts, hidden]
            curr_task_emb = task_embed[i, task_ids, :]
            curr_usv_emb = usv_embed[i, usv_ids, :]

            # [n_acts, 2*hidden] -> Action Vectors
            action_vectors = torch.cat([curr_task_emb, curr_usv_emb], dim=-1)

            # B. 提取状态向量
            # [1, 2*hidden] -> State Vector
            state_vector = global_embed[i].unsqueeze(0)

            # C. 拼接序列: [State, Action_1, Action_2, ...]
            # Shape: [1 + n_acts, 2*hidden]
            sequence = torch.cat([state_vector, action_vectors], dim=0)

            # --- 自注意力更新 ---
            # MultiheadAttention 需要输入形状: [Seq_Len, Batch, Dim]
            # 这里 Batch=1
            seq_in = sequence.unsqueeze(1)  # [1 + n_acts, 1, 2*hidden]

            # attn_output: [1 + n_acts, 1, 2*hidden]
            attn_output, _ = self.attention(query=seq_in, key=seq_in, value=seq_in)

            # 残差连接 + 层归一化
            # (Transformer Block 的标准操作)
            seq_updated = self.layer_norm(seq_in + attn_output)
            seq_updated = seq_updated.squeeze(1)  # [1 + n_acts, 2*hidden]

            # --- 拆分更新后的向量 ---
            updated_state = seq_updated[0]  # [2*hidden]
            updated_actions = seq_updated[1:]  # [n_acts, 2*hidden]

            # --- 计算 Actor 分数 ---
            # 将 "更新后的状态" 广播并拼接到 "更新后的动作"
            # [n_acts, 2*hidden]
            state_broadcast = updated_state.unsqueeze(0).expand(len(actions), -1)

            # Actor Input: [n_acts, 4*hidden]
            actor_input = torch.cat([updated_actions, state_broadcast], dim=-1)

            # 计算分数
            scores = self.actor(actor_input).squeeze(-1)
            batch_logits.append(scores)

        return batch_logits, values

    def get_action(self, state_dict, available_actions, deterministic=False):
        """
        单步推理选择动作
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
    def __init__(self, hidden_dim=64, hgnn_layers=2, lr=3e-4,
                 gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化网络
        self.policy = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        self.policy_old = ActorCritic(hidden_dim, hgnn_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.buffer = []

    def select_action(self, env, state_dict, deterministic=False):
        available_actions = env.get_available_actions()  # 返回 ID 列表

        # 将 ID 转为 (task, usv) 对
        action_pairs = [(a // env.n_usvs, a % env.n_usvs) for a in available_actions]

        state_tensor = self._prepare_state(state_dict)  # batch=1

        # 使用 policy_old 进行采样
        action_idx_in_list, log_prob, value = self.policy_old.get_action(
            state_tensor, action_pairs, deterministic
        )

        if action_idx_in_list is None:
            return None, None, None, None, None

        # 选中的动作 ID
        real_action_id = available_actions[action_idx_in_list]

        return real_action_id, action_idx_in_list, log_prob, value, action_pairs

    def _prepare_state(self, state_dict, batch=False):
        if not batch:
            return {
                'usv_features': torch.FloatTensor(state_dict['usv_features']).unsqueeze(0).to(self.device),
                'task_features': torch.FloatTensor(state_dict['task_features']).unsqueeze(0).to(self.device)
            }
        else:
            usv_feats = torch.FloatTensor(np.array([s['usv_features'] for s in state_dict])).to(self.device)
            task_feats = torch.FloatTensor(np.array([s['task_features'] for s in state_dict])).to(self.device)
            return {
                'usv_features': usv_feats,
                'task_features': task_feats
            }

    def store_transition(self, state, avail_actions, action_idx, reward, log_prob, done, value):
        self.buffer.append({
            'state': state,
            'avail_actions': avail_actions,
            'action_idx': action_idx,
            'reward': reward,
            'log_prob': log_prob.item(),
            'done': done,
            'value': value.item()
        })

    def update(self, n_epochs=4, batch_size=32):
        if len(self.buffer) == 0: return {}

        # 1. 计算 GAE
        rewards = [t['reward'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]
        values = [t['value'] for t in self.buffer]

        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(self.buffer))):
            mask = 1 - dones[i]
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * 0.95 * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备数据
        states_list = [t['state'] for t in self.buffer]
        avail_actions_list = [t['avail_actions'] for t in self.buffer]
        old_action_idxs = torch.tensor([t['action_idx'] for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.buffer], dtype=torch.float32, device=self.device)

        # 2. PPO 更新
        dataset_size = len(self.buffer)
        indices = np.arange(dataset_size)
        total_loss = 0

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_states_list = [states_list[i] for i in mb_idx]
                mb_avail_actions = [avail_actions_list[i] for i in mb_idx]
                mb_action_idxs = old_action_idxs[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                mb_state_tensor = self._prepare_state(mb_states_list, batch=True)

                # 前向传播 (含 Attention)
                batch_logits, batch_values = self.policy.forward(mb_state_tensor, mb_avail_actions)

                curr_log_probs = []
                curr_entropy = []

                for i, logits in enumerate(batch_logits):
                    if logits.numel() == 0:
                        curr_log_probs.append(torch.tensor(0.0, device=self.device))
                        curr_entropy.append(torch.tensor(0.0, device=self.device))
                        continue

                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)

                    curr_log_probs.append(dist.log_prob(mb_action_idxs[i]))
                    curr_entropy.append(dist.entropy())

                curr_log_probs = torch.stack(curr_log_probs)
                curr_entropy = torch.stack(curr_entropy).mean()
                batch_values = batch_values.squeeze(-1)

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

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {'loss': total_loss / n_epochs}

    def save(self, path):
        torch.save(self.policy.state_dict(), path)