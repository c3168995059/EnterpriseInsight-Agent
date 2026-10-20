# EnterpriseInsight-Agent

基于机器学习与大语言模型的工业设备故障预测与智能分析项目。

## 1. 项目简介

本项目面向工业设备预测性维护场景，结合机器学习故障预测、FastAPI 服务和大语言模型 Agent，实现设备数据分析、故障风险预测与自然语言交互。

## 2. 技术栈

- Python
- Pandas / NumPy
- Scikit-learn
- FastAPI / Uvicorn
- DeepSeek API（通过 OpenAI Python SDK 调用）
- Pytest

## 3. 主要功能

- 工业设备数据探索与预处理
- 基于逻辑回归的设备故障预测
- 单条及批量预测 API
- 基于工具调用的 Agent 交互
- 多轮对话历史管理
- 自动化测试与异常处理

## 4. 项目结构

```text
EnterpriseInsight-Agent/
├── src/                         # 项目核心代码
│   ├── agent.py                 # Agent 核心逻辑
│   ├── agent_tools.py           # Agent 工具定义
│   ├── tool_executor.py         # 工具调用执行器
│   ├── conversation.py          # 多轮对话管理
│   ├── chat_cli.py              # 命令行交互入口
│   ├── logger_config.py         # 日志配置
│   ├── check_config.py          # 环境配置检查
│   ├── api.py                   # FastAPI 预测接口
│   ├── predict.py               # 模型预测逻辑
│   ├── data_analysis.py         # 数据探索分析
│   ├── data_preprocessing.py    # 数据预处理
│   ├── train_baseline.py        # 基线模型训练
│   ├── train_model.py           # 模型训练与实验
│   └── evaluate_model.py        # 模型评估
│
├── scripts/                     # Agent 功能演示
│   ├── demo_llm.py              # 大模型调用演示
│   ├── demo_tool_call.py        # 工具调用演示
│   ├── demo_agent.py            # Agent 执行演示
│   ├── demo_multi_turn.py       # 多轮对话演示
│   └── demo_auto_conversation.py # 自动管理对话历史演示
│
├── tests/                       # 自动化测试
│   ├── test_agent.py
│   ├── test_api_integration.py
│   ├── test_conversation.py
│   ├── test_logger_config.py
│   └── test_tool_executor.py
│
├── data/                        # 数据集（不纳入 Git）
├── models/                      # 训练模型（不纳入 Git）
├── figures/                     # 数据分析图表
├── docs/                        # 项目技术文档
├── notebooks/                   # 数据分析与实验笔记
├── results/                     # 实验结果（不纳入 Git）
├── requirements.txt             # Python 项目依赖
├── pytest.ini                   # Pytest 配置
├── .gitignore                   # Git 忽略规则
└── README.md                    # 项目说明文档
```

**说明：** `data/`、`models/`、`results/` 等目录主要保存本地数据和实验产物，默认不提交至 Git 仓库。`docs/` 和 `notebooks/` 目前为预留目录。

## 5. 环境安装与运行

### 5.1 环境要求

- Python 3.13（当前开发环境为 Python 3.13.9）
- Git
- Windows PowerShell（以下命令以 Windows 为例）

### 5.2 创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5.3 安装项目依赖

```powershell
python -m pip install -r requirements.txt
```

### 5.4 验证安装

```powershell
python -m pip check
```

### 5.5 配置环境变量

项目通过 `.env` 文件读取 DeepSeek API 配置。

在项目根目录创建 `.env`，配置以下变量：

```dotenv
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=your_base_url_here
DEEPSEEK_MODEL=your_model_name_here
```

请使用实际服务对应的配置值，不要将真实 API Key 提交到 Git 仓库。

### 5.6 启动 FastAPI 服务

在项目根目录执行：

```powershell
python -m uvicorn api:app --app-dir src --reload
```

默认服务地址：

```text
http://127.0.0.1:8000
```

可通过以下地址查看交互式 API 文档：

```text
http://127.0.0.1:8000/docs
```

提供的主要接口：

- `GET /health`：健康检查
- `POST /predict`：单条设备故障预测
- `POST /predict/batch`：批量设备故障预测

**注意：** 预测接口需要本地已训练的模型文件。模型权重未纳入 Git 仓库，需要提前准备。

### 5.7 启动 Agent 命令行

保持 FastAPI 服务运行，打开另一个终端，激活项目虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

启动 Agent：

```powershell
python .\src\chat_cli.py
```

Agent 支持自然语言交互、工具调用及多轮对话管理。

使用前需确保 `.env` 已正确配置 DeepSeek API，并且本地 FastAPI 服务已启动。

**注意：** 实际调用 DeepSeek API 可能产生 Token 消耗。

## 6. 模型训练与评估

### 6.1 预测任务

本项目面向工业设备预测性维护场景，将设备故障检测建模为二分类任务。

- `0`：设备正常（Normal）
- `1`：设备故障（Failure）

模型使用六项设备运行特征作为输入：产品类型、空气温度、工艺温度、转速、扭矩和工具磨损时间。

### 6.2 数据预处理

项目对输入特征进行分类处理：

- 类别特征：`Type`
- 数值特征：空气温度、工艺温度、转速、扭矩、工具磨损时间

训练代码通过 Scikit-learn 的预处理组件构建特征处理流程，并使用 `stratify` 进行分层数据划分，以尽可能保持不同数据子集中的故障比例。

### 6.3 基线模型

使用逻辑回归（Logistic Regression）作为设备故障预测的基线模型。

该模型结构相对简单、计算成本较低，适合作为后续改进方法的比较基准。

### 6.4 评估指标

由于工业设备故障数据存在类别不平衡问题，本项目不仅使用 Accuracy，还综合评估：

- **Precision**：预测为故障的样本中，真实故障所占的比例。
- **Recall**：真实故障样本中，被模型正确识别的比例。
- **F1-score**：Precision 与 Recall 的调和平均值。
- **Confusion Matrix**：通过 TP、FP、TN、FN 分析模型的漏报和误报。

### 6.5 类别权重与阈值实验

为分析类别不平衡对故障检测的影响，项目实现了不同类别权重配置的对比实验。

同时，在验证集上比较以下分类阈值：

`0.05、0.10、0.20、0.30、0.40、0.50`

不同阈值会改变故障检测的 Precision 和 Recall，需要结合具体业务对漏报和误报的容忍程度进行权衡。

验证集上的实验结果用于模型选择，不能直接等同于最终测试集性能。

### 6.6 实验结果

#### 默认逻辑回归模型：测试集评估

采用逻辑回归作为基线模型，使用 80% 数据训练、20% 数据测试。数据划分使用 `random_state=42` 和分层抽样，模型采用默认类别权重和默认分类阈值 `0.5`。

| 评估指标 | 测试集结果 |
|---|---:|
| Accuracy | 0.9675 |
| Precision | 0.6364 |
| Recall | 0.1029 |
| F1-score | 0.1772 |

测试集混淆矩阵如下：

| 真实标签 / 预测标签 | 正常（0） | 故障（1） |
|---|---:|---:|
| 正常（0） | 1928 | 4 |
| 故障（1） | 61 | 7 |

**实验分析：**

- 测试集包含 2,000 条记录，其中 68 条为真实故障样本。
- 模型整体准确率达到 96.75%，但故障 Recall 只有 10.29%。
- 在 68 条真实故障记录中，模型仅正确识别 7 条，漏报 61 条。
- 由于数据存在类别不平衡，仅使用 Accuracy 无法充分评价故障检测效果。

以上结果通过 `scripts/reproduce_baseline_test.py` 独立复现得到。

**说明：** 此处为默认模型的测试集结果；下方不同阈值的对比为独立训练设置下的验证集实验结果，不应将两者直接视为同一模型的性能对比。

#### 验证集预测阈值对比

使用逻辑回归模型，在验证集上比较不同分类阈值下的 Precision、Recall 和 F1-score。

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.05 | 0.1644 | 0.7206 | 0.2678 |
| 0.10 | 0.2645 | 0.6029 | 0.3677 |
| **0.20** | **0.4000** | **0.4412** | **0.4196** |
| 0.30 | 0.5000 | 0.3088 | 0.3818 |
| 0.40 | 0.6000 | 0.2206 | 0.3226 |
| 0.50 | 0.7647 | 0.1912 | 0.3059 |

**实验分析：**

- 阈值为 `0.20` 时，在所比较的六个阈值中取得最高验证集 F1-score（0.4196）。
- 阈值为 `0.05` 时，Recall 达到 0.7206，但 Precision 只有 0.1644，误报较多。
- 阈值为 `0.50` 时，Precision 达到 0.7647，但 Recall 仅为 0.1912，漏报较多。
- 分类阈值需要根据实际场景中漏报与误报的业务成本进行选择，不能只根据单一指标决定。

实验数据来自本地 `results/lr_threshold_validation.csv`。由于 `results/` 未纳入 Git 仓库，表格中的指标作为实验记录展示。

**注意：** 以上均为验证集结果，不能直接视为最终测试集性能。模型最终测试集表现需要单独报告。

## 7. 自动化测试

项目使用 Pytest 对 Agent 核心逻辑、工具执行器、对话管理和日志配置等模块进行自动化测试。

### 7.1 测试范围

| 测试模块 | 测试数量 | 测试内容 |
|---|---:|---|
| Agent | 9 | 普通回复、工具调用、异常处理、历史消息和截断响应 |
| 对话管理 | 6 | 消息记录、清空历史、历史副本及异常处理 |
| 工具执行器 | 9 | 参数校验、HTTP 错误、超时、连接失败与正常执行 |
| 日志配置 | 2 | INFO 与 DEBUG 日志级别 |
| API 集成测试 | 1 | 对实际运行的 FastAPI 服务进行接口验证 |

项目共包含 **27 个自动化测试**，其中 26 个可以在不启动 API 服务的情况下执行。

### 7.2 运行非集成测试

在项目根目录执行：

```powershell
python -m pytest tests/ -m "not integration" -v
```

已验证的测试结果：

```text
26 passed, 1 deselected
```

其中，`1 deselected` 表示主动排除了 API 集成测试，不属于测试失败。

### 7.3 运行完整测试

首先启动 FastAPI 服务：

```powershell
python -m uvicorn api:app --app-dir src --reload
```

保持服务运行，在另一个已激活虚拟环境的终端执行：

```powershell
python -m pytest tests/ -v
```

完整测试包含需要实际 FastAPI 服务的集成测试，因此运行前应确保服务正常启动，并已准备好所需的本地模型文件。

### 7.4 测试说明

单元测试主要用于验证程序逻辑和异常处理，不代表 DeepSeek API 的真实在线调用已经通过验证。

在模型文件、数据集或外部服务不可用的情况下，部分集成测试可能无法运行。

## 8. 后续开发计划

本项目已实现基于逻辑回归的设备故障预测、FastAPI 预测接口、DeepSeek Agent 工具调用、多轮对话管理和基础自动化测试。

后续计划从以下方向继续完善：

1. **项目可复现性**：补充数据集获取说明、模型训练流程和首次运行指南。
2. **Agent 工作流优化**：学习并引入 LangGraph，探索更清晰的状态管理、工具调用和错误恢复机制。
3. **故障预测模型优化**：针对类别不平衡问题，进一步比较分类阈值、类别权重及其他机器学习方法，并通过独立测试集验证。
4. **工程化能力提升**：完善异常处理、运行日志、接口测试和配置管理。
5. **项目展示与文档**：补充系统架构图、典型交互示例和实验结果说明。

以上功能属于后续计划，将根据实际开发与测试进度逐步实现。