# TCIF / M4OE — agent code review snapshot

这是为**阅读代码、提出模型性能改进建议**整理的研究代码快照。主线为 CMU-MOSEI 上的 **Temporal Context Innovation Filtering (TCIF)**，基于多模态 Shared-Specific Soft-MoE。整理日期：2026-09-22。

源码取自实际执行过完整模型和五组消融的 MOSEI 执行树；保留训练、模型、数据、损失、评估和已有测试。实验摘要覆盖 MOSEI 主结果/消融，以及 MOSI、CH-SIMS 的近期成功与失败尝试。新增独立[CH-SIMS执行分支与逐样本诊断](cross_dataset/chsims/README.md)；同期[MOSI过程诊断与执行证据](results/mosi_diagnostics_20260922/README.md)也已补入，见[范围说明](docs/PROVENANCE.md)。

最新CH-SIMS更新：门控/context八组全部完成，context=.03、seed41得到MAE=.389766的取舍点，但未联合达标；[结果说明](results/evidence/chsims_gatectx_report.md)与[完整448点](results/chsims_gatectx_full_sweeps.json)已同步。主结果仍按测试集checkpoint与测试集eta选点。

最新P0诊断已完成：[7304条逐样本记录、重建核验与校准报告](results/evidence/chsims_diagnostics_report.md)。主结果仍使用测试集checkpoint/test eta；新训练状态另列，未将提交当作完成。

**2026-09-25 MOSI G/R/S完成：** 三项各200轮、双checkpoint完整eta共42点，均未达阶段目标；S在C1同环境参考上有小幅同点改善，未超过Lab D1主候选。见[完整结果与机制审计](results/mosi_grs_20260924/README.md)。

**2026-09-27额外数据迁移：** 同构MOSEI4轮→MOSI200轮完整完成。推荐同点46.209913/.709394/84.146341/83.769175，四项均优于原Lab D1主点，但未达最新85/85二分类阶段门槛。额外MOSEI监督单列，见[完整结果](results/mosei_to_mosi_20260927/README.md)。

**2026-09-29迁移后续完成：** 降router、原S、整块TCIF重置、统一2×2和额外MOSI计算量对照全部完成。7个新目标共98点，2个新增源预算均2044步/65304暴露；没有最新阶段联合达标。分类交互项不为正，普通融合迁移另取得.701917低MAE取舍点。见[完整报告](results/transfer_followup_20260928/README.md)。

## 给接手 agent 的阅读顺序

新增[MOSI配对梯度诊断](results/mosi_gradprobe_20260923/README.md)：D1/R1既有权重上的12组只读探针，不是新的精度实验；当前没有据此确认持续梯度冲突或提高全局裁剪阈值的收益。

1. [审查任务与交付格式](docs/REVIEW_BRIEF.md)：明确性能目标，先区分事实与假设。
2. [实验结果与已有尝试](results/EXPERIMENTS.md)：包含收益、退化、失败和比较边界。
3. [架构与源码导航](docs/ARCHITECTURE.md)：从数据一路追到最终指标。
4. [实验协议与运行说明](docs/REPRODUCE.md)：读出、checkpoint、完整 eta sweep、数据和依赖约定。
5. 阅读核心源码及对应测试，再给出有代码位置和验证方案的建议。

## 快速结论

| 证据 | Acc7 (%) ↑ | MAE ↓ | Acc2non0 (%) ↑ | macro-F1non0 (%) ↑ |
|---|---:|---:|---:|---:|
| 后续门控温度 .5 主结果，固定 eta=.8 | 56.643 | .49293 | 87.672 | 86.820 |
| 同批消融 Full TCIF，固定 eta=.8 | 56.321 | .49124 | 87.094 | 86.296 |
| 同批普通上下文融合，固定 eta=.8 | 56.428 | .49270 | 87.479 | 86.648 |

这些行来自不同实验设置，第一行不能替换消融对照。结果采用项目既定的 **test-selected** 开发协议；不代表独立留出测试的无偏泛化估计，也不代表跨 seed 显著改进。

**2026-09-22 MOSI更新：** soft-label tau=.4与router temperature=.15两项均完成200轮及双checkpoint完整eta扫描。前者未改善；后者最低MAE为.730642，同点Acc7/Acc2non0/macro-F1non0为44.314869/83.689024/83.153125，未达联合目标。D1仍为分类优先主点。见[最新结果](results/EXPERIMENTS.md#mosi-2026-09-22更新软标签分布与路由温度)及[完整评估表](results/evidence/mosi_distribution_report.md)。

**新增过程证据：** [MOSI诊断包](results/mosi_diagnostics_20260922/README.md)提供D1/R1四个checkpoint的3660条脱敏数值输出、56个val/test读出重建核验、开发集受限幅度校准和误差分组。四次校准均选a=1，停止扩扫。另附[独立MOSI执行源码](snapshots/mosi_diagnostics_20260922/README.md)；[三项训练对照现已全部完成](results/mosi_diagnostics_20260922/STUDY_RESULTS.md)：新增84点评估及5490条数值输出，均未超过D1分类主点，HEAD10回归输出不变已验证。

## 目录

```text
server-code/          模型、数据集、训练、评估、损失、已有轻量测试
configs/              完整消融配置与后续主结果参考配置（路径已替换）
scripts/              五组消融/双 checkpoint eta sweep；已有协议测试
docs/                 审查任务、架构、运行口径、来源与打包取舍
results/              摘要 + 逐 checkpoint/eta 数值证据 + 历史报告摘录
cross_dataset/chsims/ CH-SIMS实际执行分支、导出/诊断与训练路径测试
tools/                不需 GPU/数据的上传包完整性检查
```

纯阅读不需要安装依赖。先运行下面的本地检查即可验证源码语法、引用和结果网格：

```bash
python tools/check_bundle.py
python -m unittest discover -s scripts -p 'test_tcif_paper_ablation.py' -v
```

实际训练需要外部数据、backbone 和 embedding cache。路径、环境和完整命令见 [REPRODUCE.md](docs/REPRODUCE.md)。这是可审查源码包，未附权重，不能只 clone 后复现论文数字。

打包后的源码已完成静态检查、3项协议测试及25项模型测试；1项外部旧参考兼容性测试跳过，见[验证记录](docs/VALIDATION.md)。

GitHub 私有仓库：[glitter4/tcif-agent-review](https://github.com/glitter4/tcif-agent-review)。本目录内容作为仓库根目录；现有实验工作区的旧 Git 历史、SSH 配置、数据、权重、论文草稿和调度脚本均未纳入。接手 agent 需要获得该私有仓库的读取权限。

最新CH-SIMS六组对照结果已完成：[182点完整eta与结果报告](results/chsims_pathstudy_20260922/REPORT.md)。DETACH_S41得到Acc5=50.109%、MAE=.390312的取舍点，严格联合目标尚未达成。已附两组detach原始输出；四组短程对照的附加原始输出待补。

## MOSI R-Drop完成结果

[输出级R-Drop三臂对照](results/mosi_rdrop_20260929/README.md)：C0复用、C1/C2各200轮；完整42 test + 42 val点，主点未通过相对C0保护线，保留取舍结果，不自动扩网格。

## MOSI口径与分组学习率结果（2026-10-01）

[完整结果与审计](results/mosi_protocol_lr_20260930/README.md)：B/文本-only完成200轮，A第168轮中断但存量权重补评完成；42测试+42验证点、867份固定预测文件双取整审计、优化器与数据顺序验收。保持test-selected，旧/最近偶数Acc7及macro/weighted F1分列。

## MOSI EMA、邻居dropout与plateau（2026-10-01）

[完整实验与机制验收](results/mosi_regularization_20261001/README.md)：三项200轮，56test+56val点；EMA同run raw配对、邻域修复/新增错误、valMAE调度重放。邻居dropout候选46.793003%/.706336913/84.756098%/84.363082%，验证MAE有取舍，未达到阶段联合目标。

## MOSI two-hot与普通融合：首阶段

[首阶段结果与验证决策](results/mosi_twohot_20261001/README.md)：V2/V3完整200轮、28test+28val；two-hot未通过预先固定val扩展门槛，不运行V4。V1/V2的seeds40/41仍在Lab5090执行，整个任务尚未结束。

## MOSI代表结果综合比较（2026-10-02）

[均衡候选、最高Acc7与MAE／极性取舍](results/mosi_best_summary_20261002/README.md)：邻居dropout均衡点46.793003%／.706336913／84.756098%／84.363082%；two-hot最高原规则Acc7为47.084548%，有其他指标退步。完整同点数值及双取整口径已列明，不固定eta、不拼接不同点，不替代各批次验收状态。

## c1 MOSI重要过程数据补充（2026-10-02）

[过程记录与核验](results/mosi_c1_process_20261002/README.md)：三项各200轮/8200更新/16200微批次，逐轮train/val/test曲线、裁剪/组LR/抽样更新、邻居mask、帧融合权重、弱重加权损失与选模元数据；保留完整42test+42val及跨环境限制。只归档已完成数据，不修改主任务或监控状态。

## MOSI 原始输入覆盖审计（2026-10-02）

[dl01只读审计](results/mosi_input_audit_20261002/README.md)：原媒体齐备；训练/验证超6秒音频占17.45%/15.28%，音频mask长度逐条一致；少量全黑人脸采样帧、文本仅1条训练样本超过128 tokens。未重建缓存或训练，12秒覆盖数字不是性能结果。

## MOSI EMOE/FINE复现与当前模型对比（2026-10-03）

[完整比较、三seed统计和复现证据](results/mosi_baseline_reproduction_20261003/README.md)：六个种子完成、18个checkpoint重载核验通过。统一NumPy Acc7和nonzero weighted/macro-F1；TCIF均衡点46.501%/.706337/84.756%/84.363%，EMOE与FINE的best-Acc7三seed均值分别46.550%/.720709/85.061%/84.684%和48.445%/.727879/83.740%/83.398%（四项为Acc7/MAE/Acc2/macro-F1）。单seed与均值、跨环境TCIF三seed、额外MOSEI监督及FINE独立实现差异均单列；文献原表不变。
