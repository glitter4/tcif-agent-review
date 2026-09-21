# 给性能审查 agent 的任务

请基于源码与已有实验，提出可验证的模型性能提升方案。优先考虑 CMU-MOSEI 的 Acc7 与 MAE，同时关注 Acc2non0 和 macro-F1non0。吞吐、显存、训练稳定性可单列；不要将工程提速直接宣称为精度提升。

先还原一次完整前向/反向/选模/评估流程，再提出建议。完整配置优先于 CLI 默认值；历史 `bert` 命名在主线实际对应 RoBERTa。旧 MToE 类仍在文件中，但不是这份配置的 active forward。

## 值得优先查证的具体问题

| 问题 | 源码入口 | 现有证据 / 所需检验 |
|---|---|---|
| TCIF 的复杂度是否值得 | `models/tcif.py` 与 `test_tcif_ablation.py` | 普通融合分类略优、TCIF MAE 更低；先做配对误差和转折/同极性分组，不预设 TCIF 全面领先 |
| 方差估计与门控是否校准 | `TemporalContextInnovationFilter.forward`、`_compute_tcif_transition_gate_loss` | 移除 gate 损害 Acc7/MAE，却可能改善二分类；检查 gate 饱和、方差范围、冲突样本、不同标签幅值 |
| 两个任务的梯度是否冲突 | 两个 task pool、reg/cls heads、context auxiliary loss | 主目标与 context 辅助项分开检查梯度范数/夹角；MOSI 已试过移除 context 辅助，不要忽略已有证据 |
| 训练选点与推理融合是否错位 | `_update_best_checkpoints`、`compute_final_prediction` | 训练 eta=.4，主表 eta=.8；非确定性轨迹导致最优 epoch 改变。先区分选点效应与表示改进 |
| MoE 的归一化/掩码是否合理 | `SharedSpecificMoELayer` | 从实际 einsum 和 softmax 维度核对数值尺度；长视觉序列、不同模态长度是否影响专家负载；不要仅凭名称套用标准公式 |
| 邻居编码能否减少重复计算 | `EmotionM4OE.forward`、dataset/cache reader | 中心及展平邻居分别执行主干。先 profile，评估去重/推理复用；训练时 dropout、梯度和文本解冻使缓存复用不一定等价 |
| 小幅提升是否只是运行噪声 | `results/evidence/mosei_reproduction_audit.md` | 同配置存在训练轨迹分叉和 epoch 变化；微小数值差不能直接归因于单个模块 |

以上是审查入口和假设，不是已经确认的缺陷。可提出更好的问题，也可明确否定这里的假设。

## 输出要求

1. 用短文还原当前实际架构与损失，指出配置启用/关闭的分支。
2. 列出至多 5 个最有价值的建议，按预期收益、证据强度、实现成本和失败风险排序。
3. 每项给出：代码文件/函数、观察、机制假设、最小改动、对照实验、指标口径、失败判据。收益数值若无实验依据请写“待验证”。
4. 独立列出可信的正确性问题与纯性能假设；明确需要更多数据才能判断的地方。
5. 给出有限预算的实验顺序。每次只改变可解释的因素，保持双 checkpoint 的完整 eta sweep，不把不同点的最优指标拼成一行。

可建议更严格的独立泛化评估，但作为另一个协议报告，不能改写现有结果的含义。历史 MOSI/CH-SIMS 结果与当前 MOSEI 源码的实现覆盖不同，不应直接当作同一套配置的跨数据集复现。
