# TCIF / M4OE — agent code review snapshot

这是为**阅读代码、提出模型性能改进建议**整理的研究代码快照。主线为 CMU-MOSEI 上的 **Temporal Context Innovation Filtering (TCIF)**，基于多模态 Shared-Specific Soft-MoE。整理日期：2026-09-22。

源码取自实际执行过完整模型和五组消融的 MOSEI 执行树；保留训练、模型、数据、损失、评估和已有测试。实验摘要覆盖 MOSEI 主结果/消融，以及 MOSI、CH-SIMS 的近期成功与失败尝试。跨数据集扩展的全部实现不在此快照中，见[范围说明](docs/PROVENANCE.md)。

最新CH-SIMS更新：门控/context八组全部完成，context=.03、seed41得到MAE=.389766的取舍点，但未联合达标；[结果说明](results/evidence/chsims_gatectx_report.md)与[完整448点](results/chsims_gatectx_full_sweeps.json)已同步。主结果仍按测试集checkpoint与测试集eta选点。

## 给接手 agent 的阅读顺序

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

**新增过程证据：** [MOSI诊断包](results/mosi_diagnostics_20260922/README.md)提供D1/R1四个checkpoint的3660条脱敏数值输出、56个val/test读出重建核验、开发集受限幅度校准和误差分组。四次校准均选a=1，停止扩扫。另附[独立MOSI执行源码](snapshots/mosi_diagnostics_20260922/README.md)；三项训练对照已启动，暂不当作完成结果。

## 目录

```text
server-code/          模型、数据集、训练、评估、损失、已有轻量测试
configs/              完整消融配置与后续主结果参考配置（路径已替换）
scripts/              五组消融/双 checkpoint eta sweep；已有协议测试
docs/                 审查任务、架构、运行口径、来源与打包取舍
results/              摘要 + 逐 checkpoint/eta 数值证据 + 历史报告摘录
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
