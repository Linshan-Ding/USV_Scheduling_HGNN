# ============================================================================
# 5. ppo.py - PPO强化学习算法
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgnn import HGNN
from mlp import MLP
from env import USVSchedulingEnv
from instance_generator import InstanceConfig, InstanceGenerator

class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, hidden_dim=8, hgnn_layers=3):
        super().__init__()

        self.hgnn = HGNN(hidden_dim=hidden_dim, n_layers=hgnn_layers, n_neighbors=2)

        # Actor网络：输出动作概率
        # 输入：[task_embed, usv_embed, global_embed]
        actor_input_dim = hidden_dim * 3
        self.actor = MLP(
            input_dim=actor_input_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

        # Critic网络：输出状态价值
        critic_input_dim = hidden_dim * 2  # global_embed
        self.critic = MLP(
            input_dim=critic_input_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='elu'
        )

    def forward(self, state_dict, available_actions):
        """
        前向传播

        Args:
            state_dict: 状态字典
            available_actions: 可用动作列表 [(task_id, usv_id), ...]
        """
        # HGNN编码
        embeddings = self.hgnn(state_dict)

        usv_embed = embeddings['usv_embed']  # [batch, n_usvs, hidden]
        task_embed = embeddings['task_embed']  # [batch, n_tasks, hidden]
        global_embed = embeddings['global_embed']  # [batch, 2*hidden]

        # 计算每个可用动作的得分
        batch_size = usv_embed.size(0)
        n_actions = len(available_actions)

        action_scores = []
        for task_id, usv_id in available_actions:
            # 拼接task, usv, global嵌入
            task_emb = task_embed[:, task_id, :]  # [batch, hidden]
            usv_emb = usv_embed[:, usv_id, :]  # [batch, hidden]

            action_input = torch.cat([task_emb, usv_emb, global_embed], dim=-1)
            score = self.actor(action_input)  # [batch, 1]
            action_scores.append(score)

        if action_scores:
            action_scores = torch.cat(action_scores, dim=-1)  # [batch, n_actions]
            action_probs = F.softmax(action_scores, dim=-1)
        else:
            action_probs = torch.zeros(batch_size, 0)

        # 状态价值
        state_value = self.critic(global_embed)  # [batch, 1]

        return action_probs, state_value

    def get_action(self, state_dict, available_actions, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            action_probs, state_value = self.forward(state_dict, available_actions)

            if len(available_actions) == 0:
                return None, None, None

            if deterministic:
                action_idx = torch.argmax(action_probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()

            log_prob = torch.log(action_probs[0, action_idx] + 1e-8)

            return available_actions[action_idx], log_prob, state_value


class PPO:
    """PPO算法实现"""

    def __init__(self, hidden_dim=8, hgnn_layers=3, lr=3e-4,
                 gamma=1.0, epsilon=0.2, value_coef=0.5, entropy_coef=0.02):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        """选择动作"""
        available_actions = env.get_available_actions()

        # 将动作ID转换为(task_id, usv_id)对
        action_pairs = [(a // env.n_usvs, a % env.n_usvs) for a in available_actions]

        # 准备state_dict为tensor
        state_tensor = self._prepare_state(state_dict)

        action_pair, log_prob, value = self.policy_old.get_action(
            state_tensor, action_pairs, deterministic
        )

        if action_pair is None:
            return None, None, None

        # 转换回动作ID
        task_id, usv_id = action_pair
        action_id = task_id * env.n_usvs + usv_id

        return action_id, log_prob, value

    def _prepare_state(self, state_dict):
        """将numpy状态转换为tensor"""
        return {
            'usv_features': torch.FloatTensor(state_dict['usv_features']).unsqueeze(0).to(self.device),
            'task_features': torch.FloatTensor(state_dict['task_features']).unsqueeze(0).to(self.device),
            'task_coords': torch.FloatTensor(state_dict['task_coords']).unsqueeze(0).to(self.device)
        }

    def store_transition(self, state, action, reward, log_prob, value):
        """存储转移"""
        self.buffer.append((state, action, reward, log_prob, value))

    def update(self, n_epochs=1):
        """更新策略"""
        if len(self.buffer) == 0:
            return {}

        # 计算折扣回报
        returns = []
        advantages = []
        R = 0

        for i in reversed(range(len(self.buffer))):
            _, _, reward, _, value = self.buffer[i]
            R = reward + self.gamma * R
            returns.insert(0, R)
            advantage = R - value.item()
            advantages.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 更新策略
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        for epoch in range(n_epochs):
            for idx, (state, action, reward, old_log_prob, old_value) in enumerate(self.buffer):
                # 重新计算动作概率和价值
                state_tensor = self._prepare_state(state)

                # 获取可用动作
                task_id = action // len(state['usv_features'])
                usv_id = action % len(state['usv_features'])

                # 简化：假设动作仍然可用
                action_pairs = [(task_id, usv_id)]

                action_probs, value = self.policy.forward(state_tensor, action_pairs)

                if action_probs.size(-1) == 0:
                    continue

                log_prob = torch.log(action_probs[0, 0] + 1e-8)

                # PPO损失
                ratio = torch.exp(log_prob - old_log_prob.detach())
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2)

                # 价值损失
                critic_loss = F.mse_loss(value.squeeze(), returns[idx])

                # 熵损失（鼓励探索）
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()

                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空buffer
        self.buffer.clear()

        n_transitions = len(self.buffer) if self.buffer else 1

        return {
            'loss': total_loss / (n_epochs * n_transitions),
            'actor_loss': total_actor_loss / (n_epochs * n_transitions),
            'critic_loss': total_critic_loss / (n_epochs * n_transitions)
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ============================================================================
# 6. train.py - 训练主程序
# ============================================================================
def train_hgnn_ppo(n_epochs=1000, test_every=10):
    """训练HGNN-PPO"""

    # 配置
    config = InstanceConfig(
        n_usvs=4,
        n_tasks=40,
        map_size=(800, 800),
        battery_capacity=400.0,
        usv_speed=5.0
    )

    # 初始化
    generator = InstanceGenerator(config)
    ppo = PPO(hidden_dim=8, hgnn_layers=3, lr=3e-4)

    # 训练循环
    train_rewards = []
    test_makespans = []

    print("开始训练 HGNN-PPO...")
    print(f"配置: {config.n_usvs} USVs, {config.n_tasks} 任务")

    for epoch in range(n_epochs):
        # 生成新实例
        instance = generator.generate(seed=epoch)
        env = USVSchedulingEnv(instance)

        # 训练episode
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, log_prob, value = ppo.select_action(env, state, deterministic=False)

            if action is None:
                break

            next_state, reward, done, info = env.step(action)

            ppo.store_transition(state, action, reward, log_prob, value)

            episode_reward += reward
            state = next_state
            steps += 1

            if steps > config.n_tasks * 2:  # 防止无限循环
                break

        # 更新策略
        if len(ppo.buffer) > 0:
            loss_info = ppo.update(n_epochs=1)

        train_rewards.append(episode_reward)

        # 测试
        if (epoch + 1) % test_every == 0:
            test_instance = generator.generate(seed=99999)
            test_env = USVSchedulingEnv(test_instance)
            test_makespan = test_policy(ppo, test_env)
            test_makespans.append(test_makespan)

            avg_reward = np.mean(train_rewards[-test_every:])
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Test Makespan: {test_makespan:.2f}")

    return ppo, train_rewards, test_makespans


def test_policy(ppo, env, deterministic=True):
    """测试策略"""
    state = env.reset()
    done = False
    steps = 0

    while not done:
        action, _, _ = ppo.select_action(env, state, deterministic=deterministic)

        if action is None:
            break

        state, reward, done, info = env.step(action)
        steps += 1

        if steps > env.n_tasks * 2:
            break

    return info.get('makespan', float('inf'))


# ============================================================================
# 7. main.py - 主程序入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("USV调度 HGNN-DRL 系统")
    print("论文复现代码")
    print("=" * 60)

    # 训练
    trained_ppo, rewards, makespans = train_hgnn_ppo(
        n_epochs=200,
        test_every=10
    )

    print("\n训练完成！")
    print(f"最终测试 Makespan: {makespans[-1]:.2f}")

    # 保存模型
    trained_ppo.save("hgnn_ppo_model.pth")
    print("模型已保存至: hgnn_ppo_model.pth")

    # 可视化（如果有matplotlib）
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(rewards)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        ax2.plot(makespans)
        ax2.set_title('Test Makespan')
        ax2.set_xlabel('Test Episode')
        ax2.set_ylabel('Makespan')

        plt.tight_layout()
        plt.savefig('training_results.png')
        print("训练结果已保存至: training_results.png")
    except ImportError:
        print("未安装matplotlib，跳过可视化")