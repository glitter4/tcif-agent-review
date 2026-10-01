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

## CH-SIMS机制诊断增补

新增cross_dataset/chsims独立执行分支与结果目录results/chsims_diagnostics_20260922，不覆盖MOSEI主线。数据来自本轮C1只读checkpoint导出与既有日志，不重训B0/C003；包含7304条val/test原始预测、224点重建核验、1120点幅度校准、4组逐轮记录。路径脱敏，未含权重/音视频/凭据。此前“无逐样本输出/CH-SIMS执行实现”的范围说明由此处更新；MOSI同期增补范围以其过程诊断说明为准。新梯度路径训练的完成状态单列。

## MOSI只读梯度诊断增补

`results/mosi_gradprobe_20260923/`来自Lab5090既有D1/R1权重、固定128条训练样本的12条件只读探针，共48个有效batch；原训练样本ID用稳定顺序ID替换，映射仅保存在本地工作区。`snapshots/mosi_gradprobe_20260923/`提供本次探针的实际干预、训练器拦截及调度脚本，机器路径改占位。首次R1后期因显存竞争失败无有效结果，只有该阶段缺失的三项进入独立恢复目录；未重复成功结果。未导出权重、原始训练文本/音视频、缓存或历史optimizer状态，不把局部探针解释为训练收益。

## MOSI三项机制训练完成增补

`results/mosi_diagnostics_20260922/study_results.json`来自Lab任务DETACH/HEAD10/FULL10的test/val完整评估、独立输出重建与实际梯度记录，三项均exit_code0。新增12个val/test JSONL共5490条，与既有D1/R1采用同一顺序ID映射；未发布原ID映射。采集脚本及独立标准库检查器一同提供。原始指标逐值保留，移除旧runner旧目标下的best_joint/near_pass等不适用摘要，防止与本轮目标混淆。CH-SIMS记录原样保留。

## CH-SIMS pathstudy结果增补

`results/chsims_pathstudy_20260922/summary.json`来自本地`tcif_chsims_pathstudy_c1_20260922/summary.json`（C1汇总349988），保留六组182点。原始输出10个JSONL取自Lab5090 GPU恢复包，含5个checkpoint的val/test、4565条记录；附18个CPU/GPU预测比较记录及两组冻结验证。只替换机器路径，数值逐值保留；四组短程对照原始输出明确pending，未上传权重或音视频。

## MOSI G/R/S完成增量

`results/mosi_grs_20260924/`来源于C1任务350778三项200轮训练的result.json、checkpoint元数据及grs_steps.jsonl。保留全部42点与24600次更新记录；私有路径替换占位。`c1_D1_reference.json`来自旧C1 D1恢复结果而非Lab。`snapshots/mosi_grs_20260924/`包含实际干预模块、训练器及精确小补丁、机制测试与验收脚本；无网络结构改变。测试脚本导入路径作可移植调整，训练器按包约束移除字典散列种子设置。未包含权重、样本数据、缓存或凭据。

## 2026-09-27同构MOSEI→MOSI完成增量

results/mosei_to_mosi_20260927来自Lab独立迁移任务：固定源4轮val轨迹及checkpoint元数据、795张量严格迁移审计、目标200轮及14点评估、数据/cache检查。snapshots/mosei_to_mosi_20260927为对应训练器增量及默认关闭的transfer逻辑。路径脱敏，测试导入路径适配包布局；未上传模型权重、原数据、cache或任何内容摘要。源数据额外监督单列，原视频ID级检查不能证明内容级无转载。

## 迁移后四组与计算量对照增补

results/transfer_followup_20260928来自Lab独立任务的七项目标及两源阶段，含完整98点、初始化审计、router相对源漂移、源/目标固定验证诊断、旧模型逐样本配对修复聚合、精确更新/暴露/输入规模和训练墙钟。C_ST_reused.json明确复用前轮结果；2×2不使用中断的旧D1充当完整C-T。snapshots/transfer_followup_20260928是对应执行增量，普通融合从已有实现移植并按MOSI context_aux=0适配活跃参数匹配。所有路径脱敏，无权重/cache/原始样本数据；数值只对不可用非有限诊断转null，转换列表在availability.json。本批核心结果无非有限值，不修改有限指标。

## MOSI R-Drop结果增补

results/mosi_rdrop_20260929来自Lab已完成C1/C2及复用C0。42测试点、42验证点，200轮/臂与16200微批次/臂完整trace、严格795张量源初始化。逐批仅数据集位置索引，不含真实ID。机器路径替换占位；有限数值未改。snapshots/mosi_rdrop_20260929包含实际训练器、R-Drop及transfer支持模块、精确干预补丁和机制测试；测试导入适配目录、移除字典随机种子环境设置。无权重/缓存/原数据。C0历史采样顺序仅由源码/配置重建，无原逐批ID日志。

## MOSI protocol/LR增补

results/mosi_protocol_lr_20260930来源于Lab三项授权实验、其完整优化器/微批次日志及存量CSV的867份固定点指标审计（Lab727，dl01仅MOSEI消融140）。保存42新test和42val点；A SIGSEGV11、167完整轮/6886更新，补评不改变训练失败状态。B/TEXT各200轮/8200更新。只发布位置索引和类别变化，不含真实样本ID/文本/媒体/权重/cache。路径脱敏，input_audit原媒体不可用时原计数占位0明确改为null，并保留缺失数量；训练指标有限值原样保存。source快照来自实际执行树，测试/脚本路径为snapshot布局作适配，算法不变。原source剩余模块沿用随附基线；独立UpdateAudit机制测试验证不扰动参数与RNG。

## MOSI regularization增补

results/mosi_regularization_20261001来自Lab三项200轮独立训练、EMA同run原始/平均权重的各双checkpoint、完整56test+56val、优化器/顺序/mask/EMA/调度trace与位置对齐的分支诊断聚合。邻居分支原始预测仅用于离线核验，公开包不含逐样本标签/logits、原ID、权重/cache/媒体；公开微批次仅dataset位置索引。所有有限数值保留，机器路径占位替换。snapshots/mosi_regularization_20261001保存执行核心与机制测试，测试导入适配snapshot目录。严格初始化与EMA恢复由执行断言、轨迹记录及机制测试共同支持，并非保存每步权重后重算全部EMA的独立审计。

## MOSI two-hot首阶段增补

results/mosi_twohot_20261001来自Lab初始V2/V3的28test+28val、8200更新/16200微批次各两组、strict767/795源初始化、固定验证条件决策、目标分布数学审计和seed123原视频组bootstrap。V0/V1复用既有记录。数值逐值保留，仅机器路径脱敏。无权重、cache、原数据/真实ID；视频组只用位置对应的顺序整数。源码快照含two_hot、普通融合接入、分支/视频组分析和测试，测试路径适配snapshot目录。seed40/41和后续条件重加权尚未完成，不将阶段发布当作任务终结。
