# CH-SIMS 时序对比辅助监督归零：seed40/41同机结果

四组均在Lab5090独立Git工作树中从头训练50轮。C1在提交时不可达，因此每个seed先在同一张5090跑B0，再跑唯一变化为temporal_contrast_weight .0065→0的T0；这两个B0是跨机器比较必需的同机对照。TCIF、投影头、L1、hard-CE、context/gate和数据划分保持一致。

按项目test-selected协议分别保存best-Acc5/best-MAE及实际合格guarded checkpoint；每个实际保存checkpoint均完成expected/argmax七点eta。下表取各run argmax测试网格内Acc5最高、同分MAE低的代表点，各行指标来自同一checkpoint/读出/eta，百分比指标以%计。

| Run | checkpoint | epoch | eta | Acc5↑ | MAE↓ | Acc2non0↑ | F1non0↑ | Acc2(all)↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0_S40 | best_mae_model | 29 | 0.8 | 46.827 | 0.414399 | 82.474 | 81.227 | 77.243 |
| T0_S40 | best_mae_model | 30 | 0.8 | 47.265 | 0.412292 | 81.443 | 79.177 | 73.961 |
| B0_S41 | best_mae_model | 32 | 0.8 | 49.234 | 0.403544 | 80.928 | 79.739 | 76.149 |
| T0_S41 | best_guarded_acc2_model | 28 | 0.8 | 49.453 | 0.388044 | 83.505 | 82.523 | 77.462 |

## 两seed配对比较

- seed40代表点T0−B0：Acc5 +0.438个百分点，MAE -0.002108，Acc2non0 -1.031个百分点，F1non0 -2.050个百分点。
- seed41代表点T0−B0：Acc5 +0.219个百分点，MAE -0.015500，Acc2non0 +2.577个百分点，F1non0 +2.784个百分点。

完整eta点数：126。新联合目标同点合格数：0；旧50/.390/82/81目标同点合格数：0。两个指标计数来自真实同点，不拼接。

不同seed在不同物理5090上运行，同一seed的B0/T0在同一张卡上顺序运行；这仍只是两seed开发证据。C1历史结果与Lab结果有环境差异，因果判断以同机B0/T0为准。

[完整四组结果](summary.json)、[配方manifest](manifest.json)、[逐组50轮记录](runs/)、[真实执行说明](../../cross_dataset/chsims/README.md)。原始权重、音视频与凭据没有上传。

四组旧短程对照缺少的附加原始输出是独立待办；它们的训练与182个eta指标已完成。本次不将该缺项包装为已完成。
