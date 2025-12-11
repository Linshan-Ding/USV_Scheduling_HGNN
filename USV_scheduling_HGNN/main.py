# ============================================================================
# main.py - USV调度系统训练与评估主程序 (文件读取版)
# ============================================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pickle
import glob

# --- 修复 OMP Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------

from instance_generator import InstanceConfig
from env import USVSchedulingEnv
from ppo import PPO

# 创建保存目录
os.makedirs('./results', exist_ok=True)
os.makedirs('./models', exist_ok=True)


class InstanceLoader:
    """算例加载器"""

    def __init__(self, data_dir, n_usvs, n_tasks, train_ratio=0.8):
        self.pattern = os.path.join(data_dir, f"u{n_usvs}_t{n_tasks}_*.pkl")
        self.files = sorted(glob.glob(self.pattern))

        if not self.files:
            raise ValueError(f"No instance files found matching: {self.pattern}")

        # 划分训练集和测试集
        split_idx = int(len(self.files) * train_ratio)
        self.train_files = self.files[:split_idx]
        self.test_files = self.files[split_idx:]

        print(f"Loaded {len(self.files)} instances for U{n_usvs}-T{n_tasks}")
        print(f"Train: {len(self.train_files)}, Test: {len(self.test_files)}")

    def get_train_instance(self):
        """随机获取一个训练实例"""
        fpath = np.random.choice(self.train_files)
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    def get_test_instances(self):
        """获取所有测试实例"""
        instances = []
        for fpath in self.test_files:
            with open(fpath, 'rb') as f:
                instances.append(pickle.load(f))
        return instances


def evaluate(ppo_agent, test_instances):
    """
    评估函数：在固定的测试集上运行
    """
    total_rewards = []
    makespans = []

    for instance in test_instances:
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            real_action, _, _, _, _ = ppo_agent.select_action(env, state, deterministic=True)

            if real_action is None:
                break

            state, reward, done, info = env.step(real_action)
            ep_reward += reward
            steps += 1

            if steps > env.n_tasks * 3:
                break

        total_rewards.append(ep_reward)
        makespan = info.get('makespan', env.current_time) if done else 5000.0
        makespans.append(makespan)

    return np.mean(total_rewards), np.mean(makespans)


def plot_results(rewards, makespans, losses):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Average Reward per Epoch")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(makespans)
    plt.title("Avg Test Makespan")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title("PPO Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./results/training_curves.png')
    plt.close()


def main():
    # -----------------------------------------
    # 1. 配置
    # -----------------------------------------
    # 指定要训练的算例规模
    N_USVS = 4
    N_TASKS = 20
    DATA_DIR = './data/train_eval'

    # 训练超参数
    MAX_EPOCHS = 1000
    EVAL_INTERVAL = 2
    UPDATE_TIMESTEP = 2000
    LR = 3e-4

    # -----------------------------------------
    # 2. 数据加载与初始化
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        loader = InstanceLoader(DATA_DIR, N_USVS, N_TASKS)
    except ValueError as e:
        print(e)
        print("请先运行 instance_generator.py 生成数据！")
        return

    # 加载一个实例以获取维度信息（也可以直接硬编码）
    sample_inst = loader.get_train_instance()

    ppo = PPO(
        hidden_dim=64,
        hgnn_layers=2,
        lr=LR,
        gamma=0.99
    )

    history = {'rewards': [], 'test_makespans': [], 'losses': []}
    best_makespan = float('inf')
    timestep_counter = 0

    # 预加载测试集
    test_set = loader.get_test_instances()

    print("Start Training...")
    start_time = time.time()

    # -----------------------------------------
    # 3. 训练循环
    # -----------------------------------------
    for i_epoch in range(1, MAX_EPOCHS + 1):
        # 获取一个训练实例
        instance = loader.get_train_instance()
        env = USVSchedulingEnv(instance)

        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            timestep_counter += 1

            real_action, action_idx, log_prob, value, avail_acts = ppo.select_action(env, state)

            if real_action is None: break

            next_state, reward, done, info = env.step(real_action)

            ppo.store_transition(state, avail_acts, action_idx, reward, log_prob, done, value)

            state = next_state
            ep_reward += reward

            if timestep_counter % UPDATE_TIMESTEP == 0:
                update_info = ppo.update(n_epochs=4, batch_size=64)
                if update_info:
                    history['losses'].append(update_info['loss'])

        history['rewards'].append(ep_reward)

        # -----------------------------------------
        # 4. 评估
        # -----------------------------------------
        if i_epoch % 2 == 0:
            print(f"Epoch {i_epoch} | Reward: {ep_reward:.2f}")

        if i_epoch % EVAL_INTERVAL == 0:
            test_reward, test_makespan = evaluate(ppo, test_set)
            history['test_makespans'].append(test_makespan)

            print(f"--- Eval @ Epoch {i_epoch} ---")
            print(f"Test Makespan: {test_makespan:.2f}")
            print(f"Time: {time.time() - start_time:.1f}s")

            if test_makespan < best_makespan:
                best_makespan = test_makespan
                ppo.save('./models/best_model.pth')

            plot_results(history['rewards'], history['test_makespans'], history['losses'])

    print(f"Training Finished! Best Makespan: {best_makespan:.2f}")


if __name__ == "__main__":
    main()