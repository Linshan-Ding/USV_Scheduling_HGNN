# USV（无人水面艇）任务调度系统

基于强化学习的USV多智能体任务调度系统，使用PPO算法和图神经网络实现最优调度策略。

## 项目结构

```
mian_usv/
├── README.md                           # 项目说明文件（本文件）
├── env/                               # 环境模块目录（原有强化学习系统）
│   ├── __init__.py
│   ├── usv_env.py                     # USV环境核心实现
│   ├── usv_case_generator.py          # USV案例生成器
│   ├── usv_load_data.py              # USV数据加载器
│   ├── usv_constraint_validator.py     # USV约束验证器
│   ├── charging_station_manager.py     # 充电站管理器
│   └── generate_usv_validation_data.py # 验证数据生成脚本
├── graph/                             # 图神经网络模块（原有强化学习系统）
│   └── usv_hgnn.py                   # USV异构图神经网络实现
├── utils/                             # 工具模块（原有强化学习系统）
│   ├── __init__.py
│   ├── my_utils.py                   # 通用工具函数和常量
│   ├── save_manager.py              # 模型保存管理器
│   ├── model_tools.py               # 模型相关工具
│   ├── gantt_generator.py           # 甘特图生成器
│   └── color_config.json            # 颜色配置文件
├── usv_data_dev/                     # 验证数据目录（共享数据集）
│   ├── 40_2/                        # 40任务2USV组合数据
│   ├── 40_4/                        # 40任务4USV组合数据
│   ├── ...                           # 其他组合数据
│   └── 120_8/                       # 120任务8USV组合数据
├── config_usv.json                   # USV配置文件（原有强化学习系统）
├── usv_train.py                       # 训练主程序（原有强化学习系统）
├── usv_ppo.py                        # PPO算法实现（原有强化学习系统）
├── usv_mlp.py                        # MLP网络模块（原有强化学习系统）
├── usv_validate.py                   # 模型验证程序（原有强化学习系统）
├── contrast_experiment/               # 对比实验系统（新增）
│   ├── base_planner.py              # 算法基类
│   ├── utils.py                     # 工具函数
│   ├── data_adapter.py              # 数据适配器
│   ├── dispatching_rule_methods/    # 调度规则算法
│   │   ├── task_nearest_distant_first_modular.py
│   │   ├── task_farthest_distant_first_modular.py
│   │   ├── usv_lowest_battery_first_modular.py
│   │   ├── usv_highest_battery_first_modular.py
│   │   └── usv_task_random_planner_modular.py
│   ├── meta_heuristic_methods/      # 元启发式算法
│   │   ├── abc_task_planner_modular.py
│   │   ├── ga_task_planner_modular.py
│   │   └── pso_task_planner_modular.py
│   └── learning_based_methods/      # 学习算法
│       ├── dqn_task_planner_modular.py
│       └── ppo_task_planner_modular.py
├── usv_test.py                        # 统一测试框架（新增）
└── test_results/                      # 测试结果目录（新增）
```

## 核心模块功能详解

### 1. 环境模块 (env/)

#### usv_env.py - USV环境核心
**主要类：**
- **USVState**: 环境状态类，包含任务特征、USV特征、时间信息等
- **USVEvent**: 事件类，支持事件驱动的环境更新
- **USVEnv**: Gymnasium兼容的环境主类

**功能特点：**
- 支持多USV、多任务的并行调度
- 集成充电站管理，支持USV充电决策
- 实现事件驱动的状态更新机制
- 提供8大约束验证（电量、任务分配、时间约束等）
- 支持批次并行处理，提高训练效率

#### usv_case_generator.py - 案例生成器
**主要类：**
- **USVCaseData**: USV案例数据结构
- **USVCaseGenerator**: 案例生成器核心类

**功能特点：**
- 支持内存优先和文件保存两种模式
- 提供低、中、高三种随机化程度
- 支持20种预定义任务-USV组合（2/4/6/8 USV × 40/60/80/100/120 任务）
- 生成包含位置、类型、执行时间的完整任务数据
- 构建任务-USV邻接矩阵和任务前驱关系矩阵

#### usv_load_data.py - 数据加载器
**主要类：**
- **USVLoadedData**: 加载后的数据结构
- **USVDataLoader**: 数据加载器核心类

**功能特点：**
- 将USVCaseData转换为环境所需张量格式
- 构建空间邻接矩阵（任务-USV和任务-任务）
- 实现特征标准化和格式转换
- 支持单案例和批次处理

#### usv_constraint_validator.py - 约束验证器
**主要类：**
- **ConstraintResult**: 约束验证结果类
- **USVConstraintValidator**: 约束验证器核心类

**功能特点：**
- 验证8大约束条件：电池容量、任务时间、任务分配、单次出航等
- 提供详细的违反约束信息
- 支持批次验证
- 模块化设计，便于扩展新的约束类型

#### charging_station_manager.py - 充电站管理器
**主要类：**
- **ChargingRecord**: 充电记录数据结构
- **ChargingStationManager**: 充电站管理器核心类

**功能特点：**
- 支持多USV同时充电（无限充电能力）
- 维护USV充电时长和可用状态
- 提供充电统计和历史记录
- 支持充电特征提取用于决策

### 2. 图神经网络模块 (graph/)

#### usv_hgnn.py - 异构图神经网络
**主要类：**
- **USVBlock**: USV节点嵌入模块，实现"任务→USV"异构注意力更新
- **TaskBlock**: 任务节点嵌入模块，实现η近邻任务聚合更新

**功能特点：**
- 两阶段HGNN架构：Stage-1更新USV嵌入，Stage-2更新任务嵌入
- 支持单头和多头注意力机制
- 使用三元注意力：源节点特征 + 目标节点特征 + 边特征
- 完全兼容FJSP的graph/hgnn.py架构设计

### 3. 强化学习模块

#### usv_ppo.py - PPO算法实现
**主要类：**
- **Memory**: 经验存储类，适配USV二维动作格式
- **MLPs**: MLP模块整合，包含策略头和价值头
- **HGNNScheduler**: HGNN调度器，整合两阶段图神经网络
- **PPO**: PPO-Clip算法主类

**功能特点：**
- 完整的PPO-Clip算法实现
- 支持N+1动作空间（包含充电站动作）
- 集成GAE优势估计
- 支持批次训练和minibatch更新
- 提供模型保存和加载功能

#### usv_mlp.py - MLP网络模块
**主要类：**
- **USVPairFeature**: 成对特征构造器，支持N+1动作空间
- **USVActionHead**: 策略头，继承MLPActor
- **USVValueHead**: 价值头，继承MLPCritic

**功能特点：**
- 严格遵循mlp.py架构，确保训练一致性
- 支持充电站特征生成
- 使用tanh激活函数，无BatchNorm层
- 完全独立实现，不依赖外部mlp模块

### 4. 工具模块 (utils/)

#### my_utils.py - 通用工具
**功能特点：**
- 集中定义模型常量（嵌入维度、隐藏层维度等）
- 提供JSON读写工具函数
- 定义数值稳定性常量

#### save_manager.py - 模型保存管理器
**主要类：**
- **SaveManager**: 保存管理器核心类

**功能特点：**
- 配置化的实验目录创建和管理
- 最佳模型保存和管理（支持多种指标）
- 训练检查点保存和恢复
- 配置文件备份和版本管理
- 实验结果记录和查询

#### gantt_generator.py - 甘特图生成器
**主要类：**
- **GanttGenerator**: 甘特图生成器核心类

**功能特点：**
- 基于JSON数据文件生成USV任务调度甘特图
- 支持多USV并行任务可视化
- 自动颜色配置和任务标签优化显示
- 生成详细的USV工作负载汇总表格
- 提供命令行接口，便于批量处理
- 支持自定义输出路径和可视化选项

### 5. 训练和验证模块

#### usv_train.py - 训练主程序
**功能特点：**
- 完整的训练流程管理
- 支持visdom可视化训练过程
- 集成USV案例生成器和验证环境
- 支持训练日志记录和模型保存
- 提供训练统计和实验总结

#### usv_validate.py - 模型验证模块
**主要函数：**
- **get_usv_validate_env()**: 创建USV验证环境
- **usv_validate()**: USV策略验证函数

**功能特点：**
- 在训练过程中定期验证模型性能（每save_timestep步）
- 使用固定验证数据集（usv_data_dev/目录）评估性能
- 支持makespan指标计算和约束验证
- 提供验证结果统计，用于最佳模型保存决策
- 也可独立运行进行模型评估

### 6. 对比实验系统 (contrast_experiment/)

#### base_planner.py - 算法基类
**主要类：**
- **BasePlanner**: USV调度算法基类，定义统一接口
- **Task**: 任务数据类，包含位置、服务时间、优先级等属性
- **USV**: USV数据类，包含位置、电量、速度、时间线等属性

**功能特点：**
- 为所有对比算法提供统一的接口和数据结构
- 支持从JSON文件加载环境数据
- 提供基本性能指标计算功能
- 实现调度结果验证和保存功能

#### utils.py - 工具函数库
**主要功能：**
- 距离计算：欧几里得距离计算函数
- 能耗模型：简单能耗模型，考虑距离和时间因素
- 结果验证：调度结果合理性验证
- 算法比较：不同算法结果对比分析
- 数据转换：数据格式标准化和转换工具

#### data_adapter.py - 数据适配器
**主要类：**
- **USVDataAdapter**: USV数据适配器核心类

**功能特点：**
- 将不同格式的USV案例数据转换为标准格式
- 支持USV案例格式和env备份格式的自动识别
- 提供数据加载和适配的统一接口
- 确保所有算法使用相同的数据格式

#### 调度规则算法 (dispatching_rule_methods/)
**包含5个调度规则算法：**

1. **最近任务优先** (task_nearest_distant_first_modular.py)
   - 按照任务距离基地的距离从近到远排序
   - 为每个任务分配当前最空闲的USV
   - 考虑USV电量约束，必要时进行充电

2. **最远任务优先** (task_farthest_distant_first_modular.py)
   - 按照任务距离基地的距离从远到近排序
   - 优先处理距离较远的任务
   - 适用于需要避免任务遗漏的场景

3. **最低电量优先** (usv_lowest_battery_first_modular.py)
   - 优先分配任务给电量最低的USV
   - 平衡各USV的电量使用
   - 避免某些USV过度消耗

4. **最高电量优先** (usv_highest_battery_first_modular.py)
   - 优先分配任务给电量最高的USV
   - 确保复杂任务有足够电量支持
   - 适用于长距离任务较多的场景

5. **随机规划器** (usv_task_random_planner_modular.py)
   - 随机分配任务给USV
   - 作为其他算法的性能基准
   - 简单快速，无需复杂计算

#### 元启发式算法 (meta_heuristic_methods/)
**包含3个元启发式算法：**

1. **人工蜂群算法** (abc_task_planner_modular.py)
   - 模拟蜜蜂采蜜行为的优化算法
   - 通过引领蜂、跟随蜂和侦查蜂协同搜索
   - 适用于复杂的多目标优化问题

2. **遗传算法** (ga_task_planner_modular.py)
   - 基于生物进化原理的优化算法
   - 通过选择、交叉、变异操作搜索最优解
   - 适应度函数基于总完成时间和任务完成率

3. **粒子群算法** (pso_task_planner_modular.py)
   - 模拟鸟群觅食行为的群体智能算法
   - 通过粒子间的信息共享搜索最优解
   - 收敛速度快，适合实时调度场景

#### 学习算法 (learning_based_methods/)
**包含2个学习算法：**

1. **DQN算法** (dqn_task_planner_modular.py)
   - 深度Q学习算法
   - 使用神经网络逼近Q值函数
   - 支持离散动作空间的任务调度

2. **PPO算法** (ppo_task_planner_modular.py)
   - 近端策略优化算法
   - 结合了策略梯度和值函数方法
   - 训练稳定性好，适合复杂环境

### 7. 统一测试框架

#### usv_test.py - 统一测试框架
**主要类：**
- **USVTestFramework**: USV统一测试框架核心类

**功能特点：**
- 支持10种不同算法的统一测试和对比
- 自动选择代表性测试案例（20种不同规模）
- 生成详细的对比分析报告和性能排名
- 支持快速测试、标准测试和自定义测试模式
- 提供CSV、JSON和TXT格式的结果导出
- 包含成功率、完成时间、任务完成率等多维度评估

**算法支持：**
- 调度规则算法（5个）：最近任务优先、最低电量优先、最远任务优先、最高电量优先、随机规划器
- 元启发式算法（3个）：人工蜂群算法、遗传算法、粒子群算法
- 学习算法（2个）：DQN算法、PPO算法

## 文件间的联动关系

### 1. 数据流向

```
案例生成 → 数据加载 → 环境初始化 → 智能体交互 → 模型训练 → 性能验证
                    ↓
对比实验数据流：数据适配 → 算法执行 → 结果对比 → 性能分析
```

**详细流程：**
1. **usv_case_generator.py** 生成USV案例数据
2. **usv_load_data.py** 将案例数据转换为环境张量格式
3. **usv_env.py** 初始化环境状态，提供Gymnasium接口
4. **usv_ppo.py** 中的智能体与环境交互，收集经验
5. **usv_hgnn.py** 和 **usv_mlp.py** 处理状态特征，输出动作
6. **usv_train.py** 管理整个训练流程
7. **usv_validate.py** 在训练过程中定期验证模型性能

**对比实验数据流：**
1. **contrast_experiment/data_adapter.py** 适配不同格式的测试数据
2. **contrast_experiment/base_planner.py** 提供统一的算法接口
3. **各算法模块** 执行具体的调度算法
4. **usv_test.py** 统一测试和结果分析
5. **utils/gantt_generator.py** 生成调度甘特图

### 2. 核心组件协作

**环境-智能体交互：**
- `usv_env.py` 提供标准Gymnasium接口（reset、step、render）
- `usv_ppo.py` 的PPO类实现动作选择和策略更新
- `usv_constraint_validator.py` 实时验证调度决策的合法性

**特征处理流水线：**
- `usv_env.py` 的USVState类管理环境状态
- `usv_hgnn.py` 的USVBlock和TaskBlock处理图结构特征
- `usv_mlp.py` 的USVPairFeature构造成对特征
- `usv_ppo.py` 的MLPs模块输出策略和价值

**充电管理集成：**
- `usv_env.py` 集成充电站决策（N+1动作空间）
- `charging_station_manager.py` 管理充电状态和历史
- `usv_ppo.py` 增强USV特征，支持充电决策

**对比实验系统集成：**
- `contrast_experiment/data_adapter.py` 将原始数据转换为所有算法可用的标准格式
- `contrast_experiment/base_planner.py` 为所有算法提供统一的基类和接口
- `contrast_experiment/utils.py` 提供距离计算、能耗模型等通用功能
- `usv_test.py` 协调所有算法的测试执行和结果对比

### 3. 配置和工具支持

**配置管理：**
- `config_usv.json` 提供完整的配置参数
- `utils/my_utils.py` 定义模型常量
- `utils/save_manager.py` 管理实验结果和模型保存

**对比实验配置：**
- 各算法模块通过继承BasePlanner获得统一的配置管理
- `usv_test.py` 支持灵活的测试案例选择和算法配置
- 测试结果自动保存到test_results/目录

### 4. 算法间的协同关系

**强化学习算法与传统算法对比：**
- PPO算法（usv_ppo.py）与对比实验中的学习算法（DQN、PPO）形成学习算法阵营
- 调度规则算法提供快速基准性能
- 元启发式算法提供优化质量参考
- 统一测试框架确保公平比较

**数据共享机制：**
- usv_data_dev/目录中的验证数据被强化学习和对比实验系统共享
- data_adapter.py确保不同算法使用相同的数据格式
- 所有算法的结果都可以通过gantt_generator.py可视化

## 项目使用方法

### 1. 环境准备

**依赖安装：**
```bash
pip install torch gymnasium numpy pandas visdom matplotlib
```

**生成验证数据：**
```bash
cd env/
python generate_usv_validation_data.py
```

### 2. 训练模型

**启动训练：**
```bash
python usv_train.py
```

训练过程包括：
- 自动生成USV案例用于训练
- 使用PPO算法优化调度策略
- 定期验证和保存最佳模型
- 可视化训练过程（如启用visdom）

### 3. 模型验证

**在训练过程中的验证：**
- usv_train.py会定期调用usv_validate()进行性能验证
- 每`save_timestep`步验证一次当前模型
- 验证结果用于决定是否保存为最佳模型

**独立验证已训练模型：**
```bash
python usv_validate.py
```

验证功能：
- 使用固定验证数据集（usv_data_dev/目录）评估性能
- 计算makespan等关键指标
- 验证调度约束满足情况

### 4. 对比实验系统使用

#### 4.1 运行统一测试框架

**快速测试（5个案例，所有算法）：**
```bash
python usv_test.py
# 选择选项 1
```

**标准测试（20个案例，所有算法）：**
```bash
python usv_test.py
# 选择选项 2
```

**自定义测试：**
```bash
python usv_test.py
# 选择选项 3，然后选择特定算法和测试案例数量
```

#### 4.2 单独运行对比算法

**调度规则算法示例：**
```bash
cd contrast_experiment/dispatching_rule_methods/
python task_nearest_distant_first_modular.py ../../usv_data_dev/40_2/instance_01.json
```

**元启发式算法示例：**
```bash
cd contrast_experiment/meta_heuristic_methods/
python ga_task_planner_modular.py ../../usv_data_dev/40_4/instance_01.json
```

**学习算法示例：**
```bash
cd contrast_experiment/learning_based_methods/
python dqn_task_planner_modular.py ../../usv_data_dev/60_2/instance_01.json
```

#### 4.3 对比实验结果分析

测试完成后，结果将保存在`test_results/`目录中：
- `test_results_YYYYMMDD_HHMMSS.csv`: 详细测试结果
- `analysis_report_YYYYMMDD_HHMMSS.json`: 分析报告
- `comprehensive_report_YYYYMMDD_HHMMSS.txt`: 综合文本报告

**查看结果排名：**
```bash
cat test_results/comprehensive_report_*.txt
```

### 5. 甘特图生成

**生成甘特图：**
```bash
python utils/gantt_generator.py example_schedule.json
```

**命令行选项：**
```bash
python utils/gantt_generator.py example_schedule.json -o output.png --show --summary summary.png
```

**JSON数据格式：**
```json
{
  "simulation_info": {
    "makespan": 120.0,
    "num_usvs": 3,
    "min_task_time_visual": 5.0
  },
  "tasks": [
    {"task_id": 0, "position": [10, 20]},
    {"task_id": 1, "position": [30, 40]}
  ],
  "schedule_history": [
    {"usv": 0, "task": 0, "start_time": 0.0, "completion_time": 15.0},
    {"usv": 1, "task": 1, "start_time": 5.0, "completion_time": 25.0}
  ]
}
```

**功能特点：**
- 自动生成USV任务调度甘特图
- 智能任务标签显示（根据时长调整显示方式）
- 包含航行时间和工作时间的可视化
- 生成USV工作负载统计表格
- 支持自定义颜色配置和输出格式
- 兼容所有对比实验算法的输出格式

### 6. 配置说明

**主要配置项 (config_usv.json)：**

**环境参数 (env_paras)：**
- `num_tasks`: 任务数量（默认40）
- `num_usvs`: USV数量（默认4）
- `batch_size`: 批次大小（默认20）
- `map_size`: 地图尺寸（默认[800, 800]）
- `battery_capacity`: 电池容量（默认1200）
- `usv_speed`: USV航速（默认5）
- `enable_pre_validation`: 启用预验证（默认true）

**模型参数 (model_paras)：**
- `d`: 嵌入维度（默认32）
- `hidden_dim`: 隐藏层维度（默认128）
- `num_layers`: HGNN层数（默认2）
- `eta`: 任务近邻数量（默认3）

**训练参数 (train_paras)：**
- `lr`: 学习率（默认0.0002）
- `max_iterations`: 最大迭代次数（默认1000）
- `K_epochs`: PPO更新轮数（默认3）
- `gamma`: 折扣因子（默认1.0）

**保存参数 (save_paras)：**
- `base_path`: 保存基础路径（默认"./save"）
- `experiment_name`: 实验名称（默认"usv_experiment"）
- `auto_version`: 自动版本管理（默认true）

### 7. 扩展和定制

**添加新的任务类型：**
1. 修改 `usv_case_generator.py` 中的任务类型定义
2. 更新 `usv_env.py` 中的任务执行时间配置
3. 调整 `usv_ppo.py` 中的特征维度（如需要）

**调整约束条件：**
1. 在 `usv_constraint_validator.py` 中添加新的验证方法
2. 更新 `USVConstraintValidator.validate_all()` 方法
3. 修改环境参数以支持新约束

**自定义网络架构：**
1. 修改 `usv_hgnn.py` 中的图神经网络结构
2. 调整 `usv_mlp.py` 中的MLP架构
3. 更新配置文件中的维度参数

## 技术特点

### 强化学习系统
1. **模块化设计**: 环境生成、数据处理、模型训练完全解耦
2. **图神经网络**: 使用异构图神经网络处理任务-USV关系
3. **强化学习**: 基于PPO算法的端到端训练
4. **约束验证**: 实时验证8大约束条件，确保调度可行性
5. **充电管理**: 集成充电站决策，支持长时间调度
6. **可扩展性**: 支持不同规模的任务-USV组合
7. **实验管理**: 完整的实验配置、模型保存和结果记录

### 对比实验系统
8. **统一算法接口**: 所有算法继承BasePlanner基类，确保接口一致性
9. **多算法支持**: 涵盖调度规则、元启发式、学习算法三大类共10种算法
10. **数据格式适配**: 自动识别和转换不同格式的测试数据
11. **综合性能评估**: 支持成功率、完成时间、任务完成率等多维度评估
12. **批量测试能力**: 支持大规模测试案例的自动化测试
13. **结果可视化**: 集成甘特图生成，直观展示调度结果
14. **算法对比分析**: 提供详细的性能排名和对比报告

### 系统集成特性
15. **数据共享**: 强化学习和对比实验系统共享测试数据集
16. **工具复用**: 距离计算、能耗模型等工具在两套系统中通用
17. **配置统一**: 通过配置文件管理所有算法参数
18. **结果兼容**: 所有算法输出格式统一，便于对比分析
19. **扩展性强**: 支持新算法的快速集成和测试

## 性能指标

### 强化学习系统性能
- **优化目标**: 最小化最大完成时间（makespan）
- **约束满足**: 8大约束条件100%满足
- **训练效率**: 支持批次并行训练
- **泛化能力**: 支持20种不同任务-USV组合

### 对比实验系统性能
- **算法覆盖**: 支持10种不同类型算法的性能对比
- **测试规模**: 支持最大120任务、8USV的大规模测试
- **评估维度**:
  - 成功率：算法成功完成调度的比例
  - 完成时间：调度方案的总完成时间
  - 任务完成率：成功分配的任务比例
  - 执行效率：算法运行时间
- **测试覆盖**: 支持5种任务规模×4种USV数量=20种测试场景

### 系统整体性能
- **数据处理**: 自动适配多种数据格式，支持批量测试
- **结果输出**: 提供CSV、JSON、TXT三种格式的详细报告
- **可视化**: 支持调度甘特图和性能对比图表生成
- **扩展性**: 模块化设计支持新算法的快速集成

该系统为USV任务调度提供了完整的解决方案，不仅包含基于强化学习的智能调度方法，还提供了丰富的对比算法和综合评估框架，为USV任务调度研究和应用提供了强有力的技术支撑。
