# c1 MOSI三项实验：重要过程数据补充包

2026-10-02。参照此前regularization/two-hot过程包补充已有记录，仅归档、解析和核验；**没有重新训练、变更实验配置、修改主任务状态或调整自动监控**。本页是数据补充，不表示其他主任务的分析/结案步骤已完成。

## 记录范围

| 运行 | seed | 完整轮次 | optimizer更新 | microbatch | 双checkpoint epoch（Acc7 / MAE） |
|---|---:|---:|---:|---:|---|
| REWEIGHT1P5 | 123 | 200 | 8200 | 16200 | 64 / 12 |
| C1_V1_S40 | 40 | 200 | 8200 | 16200 | 8 / 58 |
| C1_V1_S41 | 41 | 200 | 8200 | 16200 | 117 / 13 |

三项共42个test点和42个val点。每个test-selected checkpoint独立完整eta={0,.2,.4,.6,.8,.9,1}，expected/T1；旧/最近偶数Acc7、all/nonzero、macro/weighted-F1全部保留。当前报告**不固定eta=.4**。训练过程中原有test选模eta=.85保留为历史事实，不能因报告方式变化而重写它。

## 已补的重要过程文件

每项 `runs/<run>/` 目录包含：

| 文件 | 用途与边界 |
|---|---|
| `config.json`、`initialization_audit.json` | 完整配置、源epoch4的795张量严格加载、未恢复optimizer |
| `optimizer_groups.json` | 各组参数名/数量与初始LR |
| `optimizer_updates.jsonl` | 全8200步全局裁剪前范数、是否裁剪、实际组LR；每轮首次更新附各组梯度/权重/实际更新范数 |
| `data_order.jsonl` | 全16200微批次的位置索引，可复核每轮1284条且无删样/重采样；不含真实样本ID |
| `neighbor_dropout.jsonl` | 全16200批原有效、屏蔽、保留邻居数及无邻居样本计数 |
| `scheduler_epochs.jsonl` | 200轮调度前后LR及验证MAE |
| `epoch_metrics.jsonl` | 从原日志提取的200轮train/val/test损失、分支指标、路由及视觉融合统计（原日志有该事件时） |
| `epoch_metrics_provenance.json` | 曲线数据的来源、精度和单位说明 |
| `lambda_history.jsonl`、`final_lambda_distribution.json` | 原实现记录的视觉center/prev/next帧混合权重；不是输出eta，也不能当TCIF有效片段邻居占比 |
| `checkpoints/*.json` | 两个已选checkpoint的精确元数据，以及test-best双选点记录 |
| `result.json`、`study_complete.json` | 完整14test+14val、同点指标来源及完成回执 |

**曲线精度注意：** `epoch_metrics.jsonl`来自日志打印值，通常仅四位小数，accuracy/F1保持日志中的0–1比例；`result.json`完整sweep中的accuracy/F1为百分数。不能用打印曲线精确重选checkpoint，精确已选记录在checkpoint metadata和selection summary中。原始日志含命令/机器路径，未公开上传。

参数更新幅度只有每轮首次optimizer step的实际抽样，不是整轮累计参数漂移；不能把全局clip=100%直接解释成某个参数组没有学习。视觉lambda保留count字段及原聚合范围，不将其误称为仅训练有效邻居的统计。

[Slurm作业与step记账](slurm_accounting.json)另外保留作业/数组标识、退出状态、elapsed、CPU、请求内存和可用MaxRSS。空值表示未提供，不能当0；job与step行不能直接相加。elapsed包含初始化、训练、评估和导出，不是纯训练计时，也不能直接与不同环境的耗时比较模型效率。

## 优化与邻居屏蔽核验

| 运行 | clip比例 | 平均裁剪前范数 | 有效邻居实际屏蔽比例 |
|---|---:|---:|---:|
| REWEIGHT1P5 | 100.000% | 24.609990 | 20.0166% |
| C1_V1_S40 | 100.000% | 24.644324 | 20.0162% |
| C1_V1_S41 | 100.000% | 24.477365 | 19.9980% |

已逐轮核验：每轮81微批次、41次更新，末批4条；位置索引每轮完整覆盖0–1283。不同seed的顺序不要求相同。scheduler记录与对应step实际LR吻合，冻结backbone在每轮抽样中active_elements及更新均为0。

## 弱重加权的实际生效记录

REWEIGHT1P5另外包含：

- `weak_weighting.json`：仅用1284条训练标签确定固定归一化，弱非零172条，平均原权重1.0669781931464175；弱样本实际权重1.405839416058394，其余.9372262773722627。
- `weak_main_batches.jsonl`：16200批的弱正、弱负、零标签计数，权重和，以及加权前后reg/cls主损失。
- 200轮均核验弱正84、弱负88、零53，权重和约1284；每批权重和符合固定训练归一化，未按batch重新归一。
- 在没有弱样本的batch，reg和cls损失都按同一非弱权重缩放，验证了不是只改分类损失。gate辅助系数仍.05，验证/测试loss保持原任务目标。

[触发条件](reweight_trigger.json)、[28个同点分组诊断](reweight_analysis.json)和[核验汇总](audit_summary.json)一并保留。分组含弱正、弱负、其余非零、零与强情感，含修复/新增错误；原Lab parent从全精度分支输出重建后，已检查旧离散指标相同、MAE差小于1e-5。没有将原始标签/logits或原预测CSV纳入本公开包。

## 本批完整sweep各自最高原规则Acc7点

| 运行 | checkpoint / epoch | eta | Acc7 % | MAE | Acc2non0 % | macro-F1non0 % |
|---|---|---:|---:|---:|---:|---:|
| REWEIGHT1P5 | best_acc7_model / 64 | 0.9 | 46.064140 | 0.720436460 | 84.451220 | 84.096289 |
| C1_V1_S40 | best_acc7_model / 8 | 0.8 | 45.918367 | 0.739875307 | 83.231707 | 82.746583 |
| C1_V1_S41 | best_acc7_model / 117 | 0.8 | 46.064140 | 0.731977126 | 83.079268 | 82.701038 |

这些是本批c1结果，不替换[已归档的综合候选比较](../mosi_best_summary_20261002/README.md)。精确值见[highest_acc7_points.json](highest_acc7_points.json)，其余取舍点保留在各run的完整sweep中，不跨点拼指标。

## 来源和解释限制

三项在c1/RTX4090、PyTorch2.1.2+cu118运行，父V1 seed123在Lab5090/PyTorch2.7.1。主任务已有明确的重跑授权与普通融合跳过决定；已归档Lab实验没有在此次同步中重跑。后续若取得原Lab40/41结果，应另存，不在原/重跑中择优覆盖。

跨环境重加权比较不能独立归因于损失权重；Lab seed123+c1 seed40/41汇总也不能视为纯seed随机性。普通融合40/41配对证据仍缺，不能用一个普通融合seed123结果复制成三组baseline。test-selected、额外MOSEI监督、样本相关性和输入媒体缺项均不因过程数据更齐而消失。

[弱重加权预检](reweight_preflight.json)、[种子重跑预检](seeds_preflight.json)、[完成回执](completion_receipts.json)及[关键实现/测试](../../snapshots/mosi_c1_process_20261002/)可追溯。数据/模型路径已脱敏；无权重、缓存、原媒体、真实ID或原始训练日志。源码快照用于审查，复现需配置实际数据与模型路径。

运行 `python tools/check_mosi_c1_process.py` 做不依赖ML的过程核验；原Torch机制测试的源码一并保存，本次没有重跑训练/GPU实验。全包检查为 `python tools/check_bundle.py`。本补充同步不修改主任务文件或监控状态。
