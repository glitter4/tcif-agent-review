# MOSI G/R/S 完成结果

2026-09-25核查。C1上三项均完整完成200轮和双test-selected checkpoint各7点eta，退出码均0。G/R/S用D1 seed123配方重新训练，分别只加router-phi投影、读出L1、弱非零符号裕量；未短续训、未组合、未新增数据。Lab5090提交时网络不可达，因此同环境比较使用C1 D1；Lab D1独立保留。

**三项均未达到阶段46.1/85/85/.730或最终>48.5/>86.95/>86.94/<.697联合目标。S的分类代表点相对C1 D1略有改善，但未超过Lab D1主点；G与R没有形成联合收益，不自动扩展组合。**

## 完整同点代表结果

A*=max(Acc2,Acc2non0)，F*=max(macro-F1(all),macro-F1(nonzero))。下表所有行A*、F*均来自nonzero列；完整数据逐点标注来源，不混入weighted-F1。expected、T=1，准确率/F1为百分数。

| 实验/用途 | checkpoint / epoch | eta | Acc7 | MAE | A* | F* |
|---|---|---:|---:|---:|---:|---:|
| C1 D1，最高Acc7参考 | best_acc7 / 40 | .9 | 43.294461 | .774613 | 82.164634 | 81.845319 |
| G，最高Acc7 | best_acc7 / 91 | 1 | 43.148688 | .764923 | 82.164634 | 81.782579 |
| G，最低MAE | best_mae / 54 | .9 | 41.545190 | .751742 | 83.231707 | 82.728248 |
| R，最高Acc7 | best_acc7 / 40 | 1 | 43.440233 | .785230 | 81.859756 | 81.471170 |
| R，最低MAE | best_mae / 92 | .9 | 41.836735 | .756494 | 82.621951 | 82.137807 |
| S，最高Acc7 | best_acc7 / 107 | 1 | 43.440233 | .760816 | 83.384146 | 82.980709 |
| S，最低MAE | best_mae / 58 | .9 | 41.107872 | .751237 | 82.926829 | 82.375744 |
| Lab D1，不同环境主点 | best_acc7 / 59 | 1 | 45.335277 | .736301 | 83.841463 | 83.356311 |

相对C1 D1最高Acc7点，S增加.145772个百分点（1/686），MAE降低约.01380，A*/F*增加约1.22/1.14个百分点。这是相同seed、同服务器的开发点比较，不是统计显著或可重复收益。C1 D1最低MAE=.752301，G/S分别仅低约.00056/.00106，且对应Acc7更低；R最低MAE反而较差。不能按指标拼接不同checkpoint/eta。

## 实际干预与验证

每项8200次optimizer更新、16200个microbatch；200个尾更新只有1个microbatch，保持原训练器accum2缩放约定。

- **G**：两层phi的联合子空间，先将回归和加权CE梯度按有效batch累积，再在全局clip前条件投影。4779/8200次（58.2805%）触发，分类近零跳过0次；逐条验证dot<0且norm²>1e-12才投影、修正系数与公式一致。非phi含router scale和原辅助梯度不变的机制测试已通过。
- **R**：保留原loss，增加.2×.5×[L1(e,y)+L1(.2 clamp(r)+.8e,y)]。所有microbatch新增加权loss均值约.038054。旧实验做过仅分类期望的SmoothL1监督，本次双读出L1仍有关联历史，不称完全全新的思想。
- **S**：仅0<abs(y)<.5，margin=min(.1,abs(y))，两读出hinge平均后按当前microbatch存在的正负组平衡，权重.05。1611/16200个microbatch无弱非零样本，对应新增loss全部为0；加权loss均值约.001911。

本批训练网络结构和推理函数没有变化；原r的L1、CE .75/tau .3、router LR2e-4/temperature .1、clip1等保持D1。验证/测试loss打印沿用原目标，新增训练项详见机制日志；不把验证loss当成已包含新项。

## 文件与复核

- [42点完整结果与配置](results.json)：保留四项原始all/nonzero/weighted字段和A*/F*来源。
- [审计摘要](audit_summary.json)：最高Acc7、最低MAE、按预先缺口尺度选择的代表点及机制计数。
- [同环境D1参考](c1_D1_reference.json)：两checkpoint完整14点；来源为旧C1恢复任务。
- `mechanism/<G|R|S>/grs_steps.jsonl`与checkpoint_metadata.json：逐有效batch日志和checkpoint轮次/GRS设置。
- [实际源码和补丁](../../snapshots/mosi_grs_20260924/README.md)：独立MOSI执行分支，默认关闭。

运行`python tools/check_mosi_grs.py`验证42点、24600个更新记录、样本数、来源列、实际epoch和机制激活；`python tools/check_bundle.py`核验全包文档与源码。原数据、权重、cache及私有路径未纳入。

G耗时3:51:47，R为3:18:21，S为3:06:25，包含本环境启动、训练和评估；不能当作通用计算成本基准。所有结果仍为单seed、test-selected开发选择，最终目标是用户参照表逐列上限，不代表与任一文献完整设置严格一致。
