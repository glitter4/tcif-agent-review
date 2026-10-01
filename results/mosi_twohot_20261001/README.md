# MOSI two-hot与普通融合：首阶段结果

2026-10-01。**这是阶段报告，整个任务尚未结束。** V2普通融合+dropout、V3 TCIF+dropout+two-hot均完整200轮/8200更新；复用V0原C0和V1已完成邻居dropout。按预先固定验证门槛，two-hot不扩展V4；V1/V2的target seeds40、41已提交，全部保留Lab5090，结果待补。

## 固定主报告点：test-best-MAE、expected/T1、eta=.4

| 臂 | epoch | Acc7旧 % | MAE | A* % | macro-F* % |
|---|---:|---:|---:|---:|---:|
| V0 | 41 | 44.314869 | 0.707390694 | 84.603659 | 84.079774 |
| V1 | 58 | 46.793003 | 0.706336913 | 84.756098 | 84.363082 |
| V2_STANDARD_DROP | 58 | 44.169096 | 0.702780029 | 85.518293 | 84.954383 |
| V3_TCIF_TWOHOT | 33 | 43.440233 | 0.726788961 | 83.841463 | 83.457044 |

本表A*/F*均来自nonzero，weighted另存。V1的Acc7更好，但普通融合V2的MAE及极性更好，不能宣称TCIF四项全面优于匹配baseline。V3主点未改善这一取舍。所有权重按test-selected旧Acc7/MAE at eta=.85选择；主报告点提前固定，仍不是独立验证集选模的确认性证据。

## Two-hot扩展判据：只读验证集诊断

| 条件 | V1 | V3 | 结果 |
|---|---:|---:|---|
| 分类期望MAE严格降低 | .741222661 | .728964122 | 通过 |
| 最终MAE严格降低 | .742578098 | .739624253 | 通过 |
| 旧Acc7下降不超过.3百分点 | 47.598253% | 44.978166% | 未通过 |
| 弱正符号错误不增加 | 5/15 | 5/15 | 持平 |
| 弱负符号错误不增加 | 6/19 | 7/19 | 未通过 |
| 弱正/弱负合计错误严格减少 | 11/34 | 12/34 | 未通过 |

完整[条件决策与原值](twohot_expansion_decision.json)在测试分支分析之前保存，判据见[预先固定manifest](manifest.json)。checkpoint自身仍来自测试选点；验证门槛不能消除这一来源偏差。后续比较固定为V1/V2，不因某个测试最好点改变分支。

## 标签构造与审计

当前距离分布tau=.3在MOSI训练1284标签上的平均绝对期望偏差=.068980753、最大=.127219069；weak-positive84条均向零压缩，weak-negative88条也均向零。two-hot线性插值的审计期望误差最大约5.55e-17。[train/val数学审计](target_bias_audit.json)不使用test标签、不代表模型增益。

two-hot仅用于MOSI目标阶段分类CE；权重.75、L1和gate .05、neighbor dropout .2均不变，源TCIF4不重训。默认distance损失逐值不变，整数/端点/质量/期望/越界拒绝及CPU/GPU梯度测试通过。普通融合复用对应B_SOURCE4、无gate及gate loss；不能把两结构称为辅助损失完全相同。

## 训练与评估验收

两项各16200微批次/8200更新，200轮；与V1的训练顺序和mask日志逐条一致。源严格加载767(standard)/795(TCIF)张量，均新optimizer。每个双checkpoint各7eta，首批总28test+28val，测试686/656、验证229/216。[验收摘要](initial_audit.json)及runs下完整日志保留。

全部分支诊断见[branch_analysis.json](branch_analysis.json)：回归、分类期望、最终融合、弱正/弱负/其余非零/零/强情感；使用原评测CSV避免概率重建舍入改变符号。标准融合方差无定义不参与核心指标，原媒体缺失仍未知，不启动缓存改造。

## seed123配对视频bootstrap（初步、条件性）

31个原视频组，10000次，seed20261001；同样的视频抽样同时用于V1/V2，比较固定best-MAE/eta.4。

| 指标 | V1−V2 | 95% percentile区间 |
|---|---:|---|
| Acc7_legacy | 2.623907 | [0.374884, 5.035998] |
| MAE | 0.003557 | [-0.013584, 0.020241] |
| Acc2_nonzero | -0.762195 | [-2.347418, 0.960072] |
| macro_F1_nonzero | -0.591301 | [-2.222828, 1.157870] |

区间条件于已选择的权重和eta，不校正测试集选择偏差、历史探索或多重比较，不能据此宣称独立显著性或多seed稳定性。[全部bootstrap数值](bootstrap_seed123_primary.json)。待追加40/41后报告全部三seed，不筛种子；源seed123固定，只估计目标训练seed波动。

## 后续范围

V1/V2 seeds40/41四项已排队/运行，用户最终要求全部Lab5090；c1没有提交训练。条件弱主损失重加权仍在后续清单，未提交；不会仅因首两项完成而结束整个任务。输入媒体与实际覆盖记录缺失，不能把未知截断率写成0。

[执行源码与测试](../../snapshots/mosi_twohot_20261001/)；无权重/cache/原媒体/真实ID/私有路径。完整14点结果分别在runs/V2_STANDARD_DROP/result.json与runs/V3_TCIF_TWOHOT/result.json，附全部val及macro/weighted/all/nonzero。工程联合门槛不是统一发表门槛，当前仍为单seed/test-selected阶段证据。
