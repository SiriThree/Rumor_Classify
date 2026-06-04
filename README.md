# 可解释的谣言检测系统

本项目对应《人工智能导论》2026 大作业，任务目标是对英文推文进行谣言检测，并输出相应的判断依据。

当前项目采用的是一套 `Transformer + RAG + 规则增强` 的混合方案：

- `Transformer` 主分类器负责输出 `0/1`
- `Hybrid Retrieval` 检索相似训练样本作为证据
- `Explanation` 模块基于预测结果和检索证据生成解释
- `Fusion` 模块在高风险样本上结合证据进行纠偏

## 当前最佳结果

当前稳定主线配置为 `configs/bertweet.yaml`，在 `val.csv` 上的最新结果为：

- 准确率：`0.8927680798004988`
- 正确数：`358 / 401`

对应方案为：

- 主模型：`vinai/bertweet-base`
- 检索：`lexical + dense hybrid retrieval`
- 决策增强：
  - `strong agreement override`
  - `evidence conflict re-check`
  - `official update resolver`

## 项目目录

```text
configs/                 配置文件
docs/                    系统方案文档
outputs/                 运行输出、评测结果、分析结果
rumer2026/               作业数据集
scripts/                 训练、评测、推理、分析脚本
src/rumor_system/        项目源码
```

## 环境安装

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要调用学校提供的大模型接口生成 explanation，请先准备 `.env`：

```bash
copy .env.example .env
```

然后在 `.env` 中填入真实接口信息。

## 快速开始

1. 训练或检查当前主模型

```bash
python scripts/train.py --config configs/bertweet.yaml
```

如果需要覆盖已有 checkpoint 重新训练：

```bash
python scripts/train.py --config configs/bertweet.yaml --force-retrain
```

2. 在验证集上评测当前主线

```bash
python scripts/evaluate.py --config configs/bertweet.yaml
```

3. 导出验证集预测结果

```bash
python scripts/infer.py --config configs/bertweet.yaml --split val
```

如果不想覆盖默认输出文件，可以指定新的输出路径：

```bash
python scripts/infer.py --config configs/bertweet.yaml --split val --output-path outputs/val_predictions_latest.csv
```

## 其他配置

### 1. 检索基线

只测试 hybrid retrieval 效果：

```bash
python scripts/evaluate.py --config configs/retrieval_hybrid.yaml
```

### 2. BERTweet 第二随机种子

```bash
python scripts/train.py --config configs/bertweet_seed7.yaml --force-retrain
python scripts/evaluate.py --config configs/bertweet_seed7.yaml
```

### 3. DeBERTa 对比实验

```bash
python scripts/train.py --config configs/deberta.yaml --force-retrain
python scripts/evaluate.py --config configs/deberta.yaml
```

说明：

- 当前 `DeBERTa` 在本项目现有设置下表现不稳定，暂时不作为主线方案
- 当前最佳结果仍来自 `BERTweet`

## 错误分析工具

导出当前配置下的错误分析摘要：

```bash
python scripts/analyze_errors.py --config configs/bertweet.yaml --output outputs/error_analysis.txt
```

给错例自动打模式标签：

```bash
python scripts/tag_error_patterns.py --predictions outputs/val_predictions.csv --output-csv outputs/error_cases_tagged.csv --output-summary outputs/error_pattern_summary.txt
```

多模型集成评测：

```bash
python scripts/evaluate_ensemble.py --config-a configs/bertweet.yaml --config-b configs/bertweet_seed7.yaml --output outputs/ensemble_predictions.csv
```

## 当前系统已实现内容

- 数据集读取与预处理
- 推文规范化
- `BERTweet` 分类器训练、保存、加载、推理
- 稀疏检索与稠密检索
- hybrid retrieval 融合
- retrieval-aware 决策融合
- 基于检索证据的 explanation prompt 构造
- 离线 explanation fallback
- 错误分析与错例模式标注脚本

## 当前主要误判类型

根据当前版本分析，剩余错误主要集中在以下几类：

- `headline_false_alarm`
  - 新闻快讯/直播类文本被误判成谣言
- `opinion_like_rumor_miss`
  - 情绪化、评论式文本中的 rumor 被误判成 non-rumor
- `evidence_conflict`
  - 检索证据和主模型预测冲突，但系统仍未完全利用正确证据
- `retrieval_supports_error`
  - 检索本身也被相似文本带偏，和主模型一起支持了错误方向

## 后续可继续优化方向

- 引入更强的 `top-k evidence reranker`
- 设计更细的多证据复判器
- 基于剩余错例继续做 hard-case 建模
- 接入学校提供的大模型接口，生成更强的 explanation

## 团队分工建议

- 成员 A：数据处理、评测脚本、实验统计
- 成员 B：Transformer 训练与调参
- 成员 C：RAG 检索、解释模块、接口接入
- 成员 D：README、报告、实验整理与复现说明
