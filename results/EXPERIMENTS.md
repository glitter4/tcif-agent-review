# 实验结果与已有尝试

以下是已有记录的整理，不是本次重新训练所得。主线 MOSEI，跨数据集仅作历史参考。准确率与 F1 为百分数，MAE 保持原量纲。除单独说明外，checkpoint/eta 都采用 test-selected 开发口径。

## MOSEI：同批五组消融

seed40、latent128、gate hidden64、batch64×accum4，训练 eta=.4。以下统一 best-Acc7 checkpoint、expected eta=.8。

| 配置 | epoch | Acc7 ↑ | MAE ↓ | Acc2non0 ↑ | macro-F1non0 ↑ |
|---|---:|---:|---:|---:|---:|
| full | 6 | 56.321 | 0.49124 | 87.094 | 86.296 |
| standard_context | 4 | 56.428 | 0.49270 | 87.479 | 86.648 |
| equal_weight | 6 | 55.999 | 0.49152 | 87.232 | 86.417 |
| no_gate | 4 | 55.463 | 0.50531 | 87.672 | 86.851 |
| shared_filter | 4 | 55.785 | 0.49345 | 87.122 | 86.387 |

完整数值：[70点 JSON](mosei_ablation_full_sweeps.json) / [CSV](mosei_ablation_full_sweeps.csv)，覆盖5配置×2 checkpoint×7 eta。实现边界与所有选择表见[原报告摘录](evidence/mosei_ablation_report.md)。

可支持的结论：去掉 gate 对 Acc7/MAE 损害较大，但二分类不一定下降；共享滤波器减少约一半滤波/上下文头参数并降低 Acc7，包含容量因素；普通融合分类略优、TCIF MAE 略低，不能写成全面领先。共享/等权/无门控都已尝试过，不宜作为全新建议直接重复。

## MOSEI：后续敏感性与当前报告主结果

| 参数 | Acc7 | MAE | Acc2non0 | macro-F1non0 |
|---|---:|---:|---:|---:|
| gate tau=.5，context weight=.1 | 56.643 | .49293 | 87.672 | 86.820 |
| gate tau=1.0，context weight=.1 | 55.827 | .49282 | 87.755 | 86.890 |
| gate tau=.75，context weight=.05 | 55.849 | .49977 | 87.204 | 86.458 |
| gate tau=.75，context weight=.20 | 55.463 | .50689 | 86.186 | 85.362 |

来源：[敏感性摘录](evidence/mosei_sensitivity_handoff.md)、[机器摘要](mosei_sensitivity_summary.json)。seed40，batch32×accum8，best-Acc7、固定 eta=.8；第一行 epoch4。这些数字为报告舍入值，未获得其他行 epoch，不猜测补填。

这一批是当时明确指定的固定读出实验，**没有完整 eta sweep**。它与消融基线使用不同机器环境及 microbatch/worker 设置；有效 batch 相同不等于训练过程严格相同。不能把第一行换入消融表后计算模块收益。

## MOSEI：复现差异与历史阶段

旧完整模型固定 eta=.8：56.536 / .49301 / 87.700 / 86.831；本轮消融完整模型同点：56.321 / .49124 / 87.094 / 86.296。训练配置相同但 deterministic=False，日志发现微小数值分叉，最终选中 epoch4 与 epoch6。证据支持轨迹与选模差异，尚未定位具体算子原因。见[复现审计](evidence/mosei_reproduction_audit.md)。

审计对两次完整模型的两个 checkpoint 各做1001点离线 eta 扫描，并验证与七个实际前向点吻合。历史最佳细扫 Acc7=56.557，重跑最佳=56.493；不应只优化完整模型而保留消融固定 eta 来扩大差距。

更早 2026-04-30 的 V7 记录中，seed43 raw regression 为 Acc7 53.77 / MAE .5235，加入 hard-CE expected-value 融合后为54.30 / .5202；soft CE 为54.33 / .5233，并伴随当时二分类下降。这只是历史路线动机，旧总表口径与当前nonzero主表不完全相同，不将这些数字作当前 TCIF 的严格增益基线。继续增大分类辅助权重当时没有带来 Acc7/MAE 收益。

## MOSI：近期尝试，不能无条件迁移到 MOSEI

D1 配方采用 latent96、最后12层文本解冻、softCE .75/tau .3、无 context auxiliary、训练 eta=.85、seed123。D1 训练中断于完成142轮后；其双 checkpoint 14点重评与旧记录在1e-5容差内匹配。当前代码包的主线并非这一跨数据集扩展实现。

| 后续尝试 / 同点 | checkpoint / epoch / eta | Acc7 | MAE | Acc2non0 | macro-F1non0 |
|---|---|---:|---:|---:|---:|
| N1：CE .75→.90，中断存量最高Acc7 | best_acc7 / 66 / .8 | 45.189504 | .747154 | 81.859756 | 81.363948 |
| N2：router LR 2e-4→5e-5，最高Acc7 | best_acc7 / 50 / .9 | 45.189504 | .754483 | 83.536585 | 83.160777 |
| N2：质量条件内最高Acc7 | best_mae / 93 / .4 | 44.897959 | .733081 | 83.689024 | 83.114543 |
| N2：最低MAE | best_mae / 93 / .6 | 44.606414 | .732663 | 83.689024 | 83.153125 |

N1完成70轮后发生CUDA内部断言错误，补评完成但不代表完成200轮训练；N2完整200轮。没有达到原联合目标46.1/.740/83.3/83.0，组合N3未触发。D1原质量合格最高Acc7 45.335277仍保留，不能用N2最低MAE来声称全面改进。

[结果说明](evidence/mosi_followup_report.md)；[D1/N1/N2的全部42点及配置](mosi_recent_full_sweeps.json)。每组都分别保留两个 checkpoint 的完整七点 expected/T=1 结果，training_complete 与 evaluation_complete 分开记录。

### MOSI 2026-09-22更新：软标签分布与路由温度

以下两项均从Lab5090 D1配方重新训练，seed123、200轮；每项只改一个字段。S1改`cls7_soft_tau .3→.4`，R1改`router_temperature .1→.15`；CE权重仍.75、router LR仍2e-4，结构与其他设置保持不变。两项完整训练及双checkpoint各7点评估均成功，无自动追加实验。

| 同点结果 | checkpoint / epoch / eta | Acc7 | MAE | Acc2non0 | macro-F1non0 |
|---|---|---:|---:|---:|---:|
| D1原主点，保留 | best_acc7 / 59 / 1 | 45.335277 | .736301 | 83.841463 | 83.356311 |
| S1 soft tau .4，最高Acc7 | best_acc7 / 58 / .6 | 43.731778 | .747584 | 82.774390 | 82.321475 |
| S1，最低MAE | best_mae / 148 / .6 | 42.274052 | .739492 | 83.536585 | 83.023913 |
| R1 router temperature .15，最高Acc7 | best_acc7 / 58 / .8 | 45.043732 | .742801 | 82.926829 | 82.469100 |
| R1，最低MAE | best_mae / 149 / .6 | 44.314869 | .730642 | 83.689024 | 83.153125 |

本次soft-label加宽没有改善已有候选；不能将单seed结果推广为所有软标签调整无效。router温度升高带来低MAE候选（比N2最低MAE再降约.00202），该点二分类质量合格，但Acc7不足。两项均未达到46.1/.740/83.3/83.0联合目标；D1保留为分类优先主点，R1单列为低MAE候选。

[完整28点报告](evidence/mosi_distribution_report.md)；[累计D1/N1/N2/S1/R1共70点与配置](mosi_recent_full_sweeps.json)；[router scale/temperature](mosi_distribution_router_scales.json)。R1的scale/temperature约7.20–7.45，温度作用未被scale完全抵消；这不能单独证明专家负载更均衡或缓解塌缩，也不能用combine求和推断均衡。

以上仍是test-selected、expected/T=1的单seed开发结果；686个全样本MAE、656个nonzero样本macro-F1，eta0仅诊断。后续泛化确认需单独采用固定开发集选点及配对多seed，不改写历史协议。新结果不扩展本仓库MOSEI源码覆盖范围。

## CH-SIMS：种子复验未重复最佳单点

四个新增 seed 均完成50轮。以下统一 best-Acc5 checkpoint（历史文件名 best_acc7_model）、argmax eta=.8、T=1；训练选 checkpoint 时使用 expected eta=1。

| seed | epoch | Acc5 | MAE | Acc2non0 | macro-F1non0 |
|---|---:|---:|---:|---:|---:|
| 41 | 31 | 47.265 | .397890 | 81.701 | 80.070 |
| 42 | 43 | 45.952 | .412877 | 80.928 | 79.261 |
| 43 | 31 | 45.077 | .407937 | 80.670 | 79.079 |
| 44 | 37 | 45.733 | .419996 | 79.124 | 77.670 |

四组所有112点最高Acc5仅47.484，未达49护栏，更未达50/.390/82/81联合目标。此前另一环境seed40的guarded epoch26参考点49.672/.399445/83.505/81.629仍保留；它采用额外受约束checkpoint规则且环境不同，不能当作只改变seed的对照。seed43曾OOM后重试成功，最终不缺评估。

[结果说明](evidence/chsims_seed_report.md)；[四seed完整112点记录](chsims_seed_sweeps.json)。当前快照不包含CH-SIMS五分类/guarded扩展，不可用MOSEI评估器直接重现这些数值。

## CH-SIMS：最新门控尺度与context权重配对实验（2026-09-22）

L1×1基线，hard-CE .3、gate loss .05、temporal .0065不变；比较gate tau=.4/.25与context auxiliary=.03，每种配方seed40/41，共8组。全部完成50轮。

**主结果按测试集checkpoint及测试集eta选点**；原test-best-Acc5、test-best-MAE及合格时的guarded候选保留。新增验证集checkpoint仅为附加记录，不参与主结果选择。完整448点中，224点为test-selected主口径、224点为val-selected附加口径；每个checkpoint独立完成expected/argmax七点eta，无缺失。8组均没有满足guard的epoch。

最有价值的是context=.03、seed41、test-best-MAE epoch24，以下每行均argmax/T=1、同checkpoint同eta：

| eta | Acc5 | MAE | Acc2non0 | macro-F1non0 | Acc2(all) |
|---:|---:|---:|---:|---:|---:|
| .8 | 48.578 | **.389766** | 82.474 | 81.061 | 76.149 |
| .9 | **48.796** | .390399 | 82.474 | 81.061 | 76.149 |

eta=.8相较同seed基线代表点，Acc5提升1.313个百分点、MAE降低.008123、Acc2non0提升.773个百分点、F1non0提升.991个百分点；但seed40相同改动退步，不能宣称稳定收益。gate tau=.4/.25没有一致联合改善。主口径全扫最高Acc5仅48.796%，仍低于49%护栏；旧50/.390/82/81联合目标也未达成。

保留该点作为低MAE取舍候选；此前另一环境的dl01 CH_L1继续作为均衡候选，不能将跨环境差异全部归因于训练配方。原B0两seed参考点与前轮同机记录一致。

[八组详情及比较表](evidence/chsims_gatectx_report.md)；[完整448点与配置](chsims_gatectx_full_sweeps.json)。网络、forward与回归损失计算未因本次实验改变；仅增加附加checkpoint记录及发现逻辑。此代码快照仍不包含CH-SIMS扩展执行树。

## 当前证据缺口

2026-09-22补入[MOSI过程诊断](mosi_diagnostics_20260922/README.md)：D1/R1逐样本输出及源码已补齐，受限尺度校准四次均选1；R1后期两路残差异号样本从77增至106，支持误差互补观察，不证明梯度冲突。已完成的旧flat-logit EMD负结果见[历史记录](evidence/mosi_ordinal_history.md)，本轮不重跑。DETACH/HEAD10/FULL10训练仍在进行，结果待补。

- 最新CH-SIMS有seed40/41小规模配对证据，但没有充分的多seed统计或独立未参与选择的测试结果。
- 没有完整的速度/显存 profile；不能从训练耗时推导算子瓶颈。
- 普通融合与TCIF尚缺按极性转折、标签幅度、邻域缺失情况的配对分析。
- MOSI D1/R1现有脱敏逐样本数值，可重建其eta指标；其他历史批次仍主要为汇总。没有逐样本文本、音视频、权重或缓存，不能据此复现全部原始模型推理。
- 历史实验不是统一预算的大规模公平排行。对于未附完整配置或源码的旧结果，只作为探索线索。
