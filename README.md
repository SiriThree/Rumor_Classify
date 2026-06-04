# 可解释的谣言检测系统

本项目对应《人工智能导论》2026 大作业，任务目标是对英文推文进行谣言检测，并输出相应的判断依据。当前系统采用 `Transformer + RAG + 规则纠偏 + 中文解释生成` 的复合方案，在保证可复现的前提下尽量提升 `val.csv` 上的检测性能。

## 当前最佳结果

当前稳定主线配置为 `configs/bertweet.yaml`，在 `val.csv` 上的最新结果为：

- 准确率：`0.9052369077306733`
- 正确数：`363 / 401`

对应方案为：

- 主模型：`vinai/bertweet-base`
- 检索：`lexical + dense hybrid retrieval`
- 决策增强：
  - `strong agreement override`
  - `evidence conflict re-check`
  - `official update resolver`
  - `rumor amplification resolver`
  - `safety notice resolver`
- 判断依据：默认输出为**中文 explanation**

## 系统框架

当前主线不是单一模型，而是一套分层系统：

1. `Transformer 主分类器`
   - 使用 `BERTweet` 对 tweet 做 `0/1` 二分类
2. `Hybrid Retrieval`
   - 从训练集检索相似推文作为证据
3. `Fusion 决策层`
   - 对高风险样本进行检索增强纠偏
4. `Explanation 模块`
   - 基于文本信号、相似证据和最终标签输出中文判断依据

## 项目目录

```text
configs/                 配置文件
docs/                    系统方案文档
outputs/                 运行输出、评测结果、分析结果
rumer2026/               作业数据集
scripts/                 训练、评测、推理、分析、报告生成脚本
src/rumor_system/        项目源码
report.docx              大作业报告（Word）
report.pdf               大作业报告（PDF）
```

## 环境安装

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要调用学校提供的大模型接口，请先准备 `.env`：

```bash
copy .env.example .env
```

然后在 `.env` 中填入真实接口信息。系统目前读取的字段为：

- `SJTU_API_BASE_URL`
- `SJTU_API_KEY`
- `SJTU_API_MODEL`

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

## 当前推荐输出文件

如果你想查看当前最好主线的中文判断依据，推荐使用：

- `outputs/val_predictions_mainline_363_zh.csv`

其中包含：

- `pred_label`
- `confidence`
- `prediction_source`
- `top_evidence_text`
- `explanation`（中文判断依据）

## 其他实验配置

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

- 当前 `DeBERTa` 在本项目设置下表现不稳定，暂时不作为主线方案
- 当前最佳结果仍来自 `BERTweet`

### 4. 事件前缀输入实验

```bash
python scripts/train.py --config configs/bertweet_event.yaml --force-retrain
python scripts/evaluate.py --config configs/bertweet_event.yaml
```

说明：

- 我们验证了显式加入 `[event: xxx]` 前缀输入
- 在当前数据上，该实验未优于稳定主线

### 5. LLM 语义分析实验线

```bash
python scripts/evaluate.py --config configs/bertweet_semantic.yaml
python scripts/infer.py --config configs/bertweet_semantic.yaml --split val --output-path outputs/val_predictions_semantic.csv
```

说明：

- 已支持 `.env` 读取、语义缓存、进度打印
- 当前真实 LLM 改判实验整体不如稳定主线，因此不作为默认方案

## 错误分析工具

导出当前配置下的错误分析摘要：

```bash
python scripts/analyze_errors.py --config configs/bertweet.yaml --output outputs/error_analysis.txt
```

给错例自动打模式标签：

```bash
python scripts/tag_error_patterns.py --predictions outputs/val_predictions.csv --output-csv outputs/error_cases_tagged.csv --output-summary outputs/error_pattern_summary.txt
```

对比主线和 semantic 实验线：

```bash
python scripts/analyze_semantic_experiment.py
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
- 中文 explanation 生成
- semantic analyzer 实验线、缓存与进度日志
- 错误分析与错例模式标注脚本
- 报告自动生成脚本

## 当前主要误判类型

根据当前版本分析，剩余错误主要集中在以下几类：

- `headline_false_alarm`
  - 新闻快讯 / 直播类文本被误判成谣言
- `opinion_like_rumor_miss`
  - 情绪化、评论式文本中的 rumor 被误判成 non-rumor
- `evidence_conflict`
  - 检索证据和主模型预测冲突，但系统仍未完全利用正确证据
- `retrieval_supports_error`
  - 检索本身也被相似文本带偏，和主模型一起支持了错误方向

## 已验证的重要结论

- 训练集和验证集覆盖相同事件集合，因此当前高分部分受益于事件内相似表达和可检索证据
- `LLM semantic analyzer` 真实接入后整体没有超过主线
- 直接加入 `event` 前缀输入也没有优于当前主线
- 当前最有效的提升来自：`BERTweet + hybrid retrieval + 面向错例的窄规则纠偏`

## 后续可继续优化方向

- 引入更强的 `top-k evidence reranker`
- 如果能拿到原始 PHEME thread，可加入 `parent/source tweet` 上下文
- 基于剩余错例继续做 hard-case 建模
- 继续提升中文判断依据的细致程度和可读性

## 报告文件

本仓库已包含大作业报告：

- [report.docx](./report.docx)
- [report.pdf](./report.pdf)

## 仓库地址

[SiriThree/Rumor_Classify](https://github.com/SiriThree/Rumor_Classify)
