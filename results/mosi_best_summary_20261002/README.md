# MOSI当前代表结果：均衡候选与单项取舍

更新：2026-10-02。本页补充已经核对的结果比较，不替代各批次完整实验报告或验收状态。

**综合兼顾Acc7、MAE和极性，优先保留TCIF迁移＋邻居dropout的均衡候选；若只追求原取整规则下的最高Acc7，则是two-hot的47.084548%。** “均衡”是对指标取舍的判断，不是一个预先定义的复合分数，也不表示该点支配所有其他结果。

## 三个不同用途的代表点

全部为seed123、Lab5090、额外MOSEI监督迁移设置；expected读出、T=1。准确率/F1为百分数，MAE为原量纲。每行四项指标来自同一checkpoint、同一eta。

| 用途 | 实验 | checkpoint / epoch | eta | Acc7原规则 | Acc7最近偶数 | MAE | Acc2non0 | macro-F1non0 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **均衡候选** | TCIF＋邻居dropout .2 | best_mae_model / 58 | .4 | **46.793003** | 46.501458 | **.706336913** | **84.756098** | **84.363082** |
| **最高原规则Acc7候选** | TCIF＋邻居dropout .2＋two-hot | best_acc7_model / 58 | .9 | **47.084548** | 46.793003 | .713692339 | 83.231707 | 82.895286 |
| 更偏MAE／极性的取舍 | 普通融合＋邻居dropout .2 | best_mae_model / 58 | .4 | 44.169096 | 43.877551 | .702780029 | 85.518293 | 84.954383 |

均衡候选的weighted-F1nonzero为 **84.748540%**；two-hot点为83.268275%，普通融合点为85.407287%。不能把macro-F1与文献weighted-F1直接当成同一列比较。

**这里没有恢复固定eta=.4的报告约定。** 邻居dropout这一点来自完整sweep，并同时取得该次运行最高原规则Acc7和最低MAE；two-hot的最高原规则Acc7在eta=.9。普通融合一行是该次运行最低MAE点，不代表所有历史实验的最低MAE。最近偶数口径单独列示，不把“最高原规则Acc7”扩展成另一口径的最高值。

## 如何解读

- two-hot相对均衡候选提高原规则Acc7约0.291545个百分点，但MAE增加约.007355，非零Acc2和macro-F1明显下降，因此不作为当前综合替代。
- 普通融合一行的MAE和极性更好，Acc7较低；这说明当前仍存在多指标取舍，不能宣称TCIF全面领先普通融合。
- 邻居dropout候选的测试表现较好，但对应验证MAE曾退步，其他运行也未证明稳定优势。其余实验的最好指标不能拼接到这个checkpoint上。
- 两个选中checkpoint均完成eta={0,.2,.4,.6,.8,.9,1}。仍采用test-selected开发协议，最近偶数只是额外评测列；取消固定报告eta不改变过去的训练或选择过程。
- 均衡候选及本页所列点均未达到阶段四指标联合目标。额外MOSEI监督、单seed结果与测试选择限制仍须保留；“有竞争力”不等于多seed稳定领先或同监督数据公平优越。

## 原始依据与完整sweep

- [邻居dropout完整14test＋14val及配置](../mosi_regularization_20261001/runs/NEIGHBOR_DROP/result.json)，[本批验收与分组报告](../mosi_regularization_20261001/README.md)。
- [two-hot完整14test＋14val及配置](../mosi_twohot_20261001/runs/V3_TCIF_TWOHOT/result.json)。其未通过预设验证扩展条件，最高测试Acc7不改变已记录的决策。
- [普通融合＋dropout完整14test＋14val及配置](../mosi_twohot_20261001/runs/V2_STANDARD_DROP/result.json)，[首阶段报告与验证决策](../mosi_twohot_20261001/README.md)。
- [本页精确数值与来源索引](selected_points.json)。本次只追加汇总，不改训练、实验状态、自动监控或主任务的后续验收记录。
- [c1三项实验的重要过程数据补充包](../mosi_c1_process_20261002/README.md)：逐步优化器/采样/mask记录、逐轮曲线、弱重加权实际损失、精确选模与完整sweep，附独立核验脚本。

## 2026-10-03复现基线对照

新增[EMOE/FINE实际复现与本页候选同口径比较](../mosi_baseline_reproduction_20261003/README.md)，包含Acc7双取整、Acc2、weighted/macro-F1、MAE、逐seed结果及跨环境限制。本页既有代表点及来源不变。
