# CMU-MOSI：EMOE / FINE 复现与当前模型对比

更新：2026-10-03。六个种子训练完成，18个目标checkpoint全部重载核验通过。此次只归档和比较，未新增训练。

**当前TCIF均衡点的MAE优于两种复现方法的best-Acc7均值；Acc7低于FINE独立实现，二分类/F1优于FINE但略低于EMOE。不存在一个方法在这些同点指标上全面领先。**

## 同口径代表点与复现均值

Acc7主比较采用NumPy最近偶数；项目half-away-from-zero另列。MAE含686条测试样本，Acc2/F1排除真实零标签、使用656条。每行均保持同一checkpoint/eta的指标；均值行是各seed对应检查点的平均，不是逐指标拼接最优。

| 结果 | Acc7 NumPy | Acc7 项目 | MAE | Acc2 nonzero | weighted-F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| TCIF 均衡点，seed123 | 46.501 | 46.793 | 0.7063 | 84.756 | 84.749 | 84.363 |
| TCIF two-hot，seed123 | 46.793 | 47.085 | 0.7137 | 83.232 | 83.268 | 82.895 |
| 普通融合，seed123 | 43.878 | 44.169 | 0.7028 | 85.518 | 85.407 | 84.954 |
| EMOE best-Acc7三seed均值 | 46.550 | 46.356 | 0.7207 | 85.061 | 85.054 | 84.684 |
| FINE best-Acc7三seed均值 | 48.445 | 48.202 | 0.7279 | 83.740 | 83.768 | 83.398 |
| EMOE 最高Acc7单seed 1111 | 47.813 | 47.230 | 0.7089 | 85.366 | 85.351 | 84.974 |
| FINE 最高Acc7单seed 8757 | 49.125 | 48.834 | 0.7208 | 84.299 | 84.317 | 83.948 |

TCIF均衡点：best_mae_model/epoch58/eta=.4；two-hot：best_acc7_model/epoch58/eta=.9；普通融合：best_mae_model/epoch58/eta=.4。均来自两个已选checkpoint各七eta的完整扫描，不恢复固定eta报告约定。参考[现有代表点说明](../mosi_best_summary_20261002/README.md)。

## 相对当前TCIF均衡点的差距

下表为“TCIF均衡点 − 基线三seed best-Acc7均值”。准确率/F1单位为百分点，MAE负值表示TCIF更低；单seed与三seed均值仅作描述性比较。

| 结果 | Acc7 NumPy | Acc7 项目 | MAE | Acc2 nonzero | weighted-F1 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| TCIF − EMOE | -0.049 | 0.437 | -0.0144 | -0.305 | -0.306 | -0.321 |
| TCIF − FINE | -1.944 | -1.409 | -0.0215 | 1.016 | 0.981 | 0.965 |

与FINE均值相比，TCIF少约1.94个百分点Acc7，但MAE低约0.02154，Acc2和两种F1更高。与EMOE均值相比，Acc7几乎持平（−0.049个百分点），MAE低约0.01437，Acc2/weighted-F1/macro-F1分别低约0.305/0.306/0.321个百分点。差值不构成统计显著性或因果结论。

## 三seed汇总：明确跨环境限制

TCIF使用已有seed123/Lab5090＋seed40/41/c1，在各run已归档14点中按NumPy Acc7最高取点；若并列，取MAE较低者，并在comparison.json保留全部并列点。原训练checkpoint选择不变。下表统一总体标准差；机器文件另附样本标准差。

| 方法 | Acc7 NumPy mean±SD | MAE mean±SD | Acc2 mean±SD | weighted-F1 mean±SD | macro-F1 mean±SD |
|---|---:|---:|---:|---:|---:|
| emoe | 46.550±1.222 | 0.7207±0.0179 | 85.061±0.777 | 85.054±0.731 | 84.684±0.699 |
| fine | 48.445±0.481 | 0.7279±0.0051 | 83.740±0.400 | 83.768±0.396 | 83.398±0.402 |
| TCIF mixed-host | 46.064±0.315 | 0.7261±0.0143 | 83.689±0.757 | 83.681±0.756 | 83.270±0.773 |

TCIF的跨环境汇总不能视为纯seed方差，也不是与两基线同种子、同输入、同监督量的公平对照。普通融合缺少已验收的多seed配对结果，不补造均值。

## 复现到什么程度

- EMOE使用[官方源码](https://github.com/fuyyyyy/EMOE)。test选优单seed Acc7达到47.813%，最低MAE为0.7062，未达到论文0.697；按官方validation-MAE选轮次的参考均值为Acc7 44.898%、MAE0.7299。两种选优协议分列。
- FINE未找到可核实官方代码；按[论文和附录](https://arxiv.org/html/2511.20167v1)独立实现。最高Acc7为49.125%，但best-Acc7均值Acc2/weighted-F1只有83.740%/83.768%，没有同时复现原表86.95%/86.94%。Q-Former/MI/调度假设和5/20对74/35的特征差异保留，不能据此证明严格复现或否定论文。
- 两基线都是单回归输出，没有eta融合定义；eta扫描不适用。TCIF仍完整保留双checkpoint各7eta。
- TCIF采用额外MOSEI监督迁移、RoBERTa/ViT/HuBERT与时序上下文；基线在MOSI上训练BERT＋MMSA特征。现有差距不能单独归因于TCIF结构。所有test选优结果均是开发口径，不是独立泛化估计。
- “当前”指仓库中已有完整评估证据的代表模型；音频6/12秒及B64等后续任务未在本包提供最终验收证据，不将进行中或仅有局部日志的结果替换进主表。

## 可审查证据

- [完整逐种子/检查点指标](RESULTS.md)、[机器摘要](summary.json)、[统一比较数据与来源](comparison.json)。
- [实验协议和实现假设](PROTOCOL.md)、[环境](environment.json)、[18个checkpoint核验](checkpoint_verification.json)。
- `runs/`保留六次运行的配置、逐轮聚合train/valid/test指标、选模记录与完成回执。预测文件名仅供本地追溯；原预测/标签/真实ID、数据、权重、缓存、原始服务器日志均不上传。
- [源码与出处](../../snapshots/mosi_baseline_reproduction_20261003/README.md)：EMOE官方模型/工具、实际运行器、FINE独立实现和已执行的关键检查。
- 当前TCIF依据：[邻居dropout](../mosi_regularization_20261001/runs/NEIGHBOR_DROP/result.json)、[two-hot](../mosi_twohot_20261001/runs/V3_TCIF_TWOHOT/result.json)、[普通融合](../mosi_twohot_20261001/runs/V2_STANDARD_DROP/result.json)、[c1 seed40](../mosi_c1_process_20261002/runs/C1_V1_S40/result.json)、[c1 seed41](../mosi_c1_process_20261002/runs/C1_V1_S41/result.json)。
- 文献原表保持不变：[literature_baselines](../literature_baselines/README.md)。本次实测与文献引用分开，不覆盖论文数值。

本地聚合证据核验：`python tools/check_mosi_baseline_comparison.py`；全包引用/语法核验：`python tools/check_bundle.py`。
