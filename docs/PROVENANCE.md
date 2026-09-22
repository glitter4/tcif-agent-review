# 来源与取舍

整理日期：2026-09-22。面向代码阅读与性能审查；没有修改原实验源码或既有实验树，没有新训练或模型选点。整理完成后，按用户要求将本包发布到独立的GitHub私有仓库；未上传原实验仓库的历史。

## 源码范围

主线为 dl01 上名为 `m4oe-tcif-paper-ablation-20260909` 的实际执行树，2026-09-22通过只读SSH获取明确列出的源码、测试和配置。核心模型/训练/评估/TCIF消融文件与本地同名执行树一致；其余文件的差异仅是换行格式。源执行树有未提交改动，所以本包根据工作区文件获取，不能用旧Git提交来代表它。

选择这个树是因为它同时具有可追溯的MOSEI结果、完整模型、四项消融和测试，比根目录旧server-code或五月快照更适合审查。它不是所有实验分支的合并，也不声称包含MOSI/CH-SIMS最新扩展。跨数据集记录作为数值参考单列。

## 打包变更

- 模型计算、损失、训练循环、评估公式及原测试保留。
- 机器用户名与绝对路径替换为 `/path/to/...`；配置输出放到 `./outputs/`。
- `output_layout.py` 与训练汇总根路径改为包内outputs，保留 `M4OE_STRUCTV5_ROOT` 环境覆盖。
- 移除运行时设置 Python 字典散列种子的语句，以遵守项目禁用散列的约束；随机数、NumPy、Torch seed设置保留。
- 添加本地 `models`/`datasets` 包标记，避免与同名第三方包冲突。
- 未使用的 magnitude postprocessing adapter 未包含，主线无对此模块的导入。
- `mosei_gate_tau05_reference.json` 是按原完整配置及敏感性runner覆写规则重建的参考文件，来源与局限见运行文档。

逐文件来源与字节数见 [source_inventory.json](source_inventory.json)。只使用直接文本/字节比较，没有生成内容摘要。

## 实验证据来源

| 包内文件 | 原工作区记录 |
|---|---|
| `results/mosei_ablation_full_sweeps.json` | `analysis/tcif_paper_ablation_results_20260915.json`，逐值保留 |
| `results/evidence/mosei_ablation_report.md` | `analysis/tcif_ablation_results_handoff_20260915.md`，去掉调度/绝对路径章节 |
| `results/evidence/mosei_reproduction_audit.md` | `analysis/tcif_reproduction_audit_20260910/REPORT.md` |
| `results/evidence/mosei_sensitivity_handoff.md` | `analysis/tcif_writing_handoff_20260913.md` |
| `results/mosi_recent_full_sweeps.json` | `analysis/tcif_mosi_d1_followup_20260920/` 中 D1_reference、all_results、N1_recovered_result，保留配置、points、epoch、状态 |
| `results/chsims_seed_sweeps.json` | `.codex-jobs/tcif_chsims_seeds_c1_20260919/summary.json`，路径替换 |
| MOSI/CH-SIMS摘录 | `results_20260921.md`、`chsims_seeds_c1_results_20260919.md` |
| 早期V7概括 | `analysis/structv7_weekly_total_table_20260430.md`，仅作历史线索 |

2026-09-22结果增量：`results/mosi_recent_full_sweeps.json`新增S1_soft_tau04与R1_router_temp015，来自`analysis/tcif_mosi_distribution_20260922/all_results.json`；逐值保留28点、checkpoint轮次、配置及完成状态，机器路径沿用包内`/path/to/`占位。`results/evidence/mosi_distribution_report.md`来自该批`results_20260922.md`，仅调整包内证据链接；`results/mosi_distribution_router_scales.json`保留两项的checkpoint尺度诊断。没有加入权重、数据、机器路径、训练源码或调度脚本。

报告为已有记录的保真摘录；文中的其他机器或数据集名称是历史背景，不是当前操作指令。

## 没有纳入的内容

2026-09-22追加MOSI过程诊断：`results/mosi_diagnostics_20260922/`来自独立Lab诊断任务的四checkpoint无梯度输出和现有日志。顺序ID代替真实sample/video ID，保留跨checkpoint连接；原映射不发布。`snapshots/mosi_diagnostics_20260922/`独立保存MOSI实际执行源代码、原训练器和本次受控干预版本；不覆盖根目录MOSEI实现。附标准库读出重建检查。旧wave38/39 EMD结果由已有日志摘录，未重新训练。

以下旧打包排除列表中的“预测文件”对本次新增的脱敏数值诊断作明确例外；仍不包含文本、原始标识、音视频、checkpoint或embedding cache。

数据集、音视频、逐样本文本、embedding cache、checkpoint、TensorBoard、原始训练日志、远程凭据/SSH配置、调度与同步脚本、临时补丁、重复快照、OASIS等探索分支、论文草稿/PDF/专利文件、第三方下载仓库、无明确结论的待运行计划。旧实验用摘要和有来源的结果代表，避免几十个版本造成阅读歧义。

原根目录README主要是HPC同步说明，旧server-code README介绍医学M4OE上游，均不适合作为当前模型入口。当前包改用专门的阅读说明，并在 [ATTRIBUTION.md](../ATTRIBUTION.md) 保留上游背景。

## 2026-09-22 CH-SIMS结果增补

`results/chsims_gatectx_full_sweeps.json`来自工作区`.codex-jobs/tcif_chsims_gatectx_c1_20260922/summary.json`，保留8组配置与448点原始指标，机器路径替换为`/path/to/...`，添加主/附加口径说明。`results/evidence/chsims_gatectx_report.md`来自`analysis/chsims_gatectx_c1_results_20260922.md`，移除机器访问与绝对路径章节。同步更新结果索引及完整性检查；未同步训练源码、权重或逐样本数据。
