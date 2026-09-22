# 三项训练对照完成结果

2026-09-22。DETACH完成200轮，HEAD10/FULL10各完成10轮，三项退出码均0，无中断或重试。三项各自test-selected双checkpoint和单列的val-selected双checkpoint均完成七点eta，共84点；新增5490条匿名逐样本记录。以下四项指标始终同一点、expected/T=1。

## 主协议：test-selected开发结果

| 实验/用途 | checkpoint / epoch | eta | Acc7 | MAE | Acc2non0 | macro-F1non0 |
|---|---|---:|---:|---:|---:|---:|
| 原D1主点，保留 | best_acc7 / 59 | 1 | 45.335277 | .736301 | 83.841463 | 83.356311 |
| DETACH最高Acc7 | best_acc7 / 81 | .8 | 44.752187 | .751174 | 83.384146 | 82.930003 |
| DETACH最低MAE | best_mae / 148 | .6 | 43.294461 | .730753 | 83.841463 | 83.282043 |
| HEAD10最高Acc7 | best_mae / 7 | .8 | 44.606414 | .735785 | 84.146341 | 83.616226 |
| HEAD10最低MAE | best_mae / 7 | .6 | 43.731778 | .735375 | 84.146341 | 83.558897 |
| FULL10最高Acc7 | best_acc7 / 9 | .9 | 44.897959 | .755533 | 83.231707 | 82.910031 |
| FULL10最低MAE | best_mae / 1 | .8 | 43.877551 | .741883 | 83.384146 | 83.028214 |

三项均未达46.1/.740/83.3/83.0联合目标，也未超过D1最高Acc7。DETACH仅形成回归取舍，其最低MAE仍略高于先前router-temperature R1的.730642。HEAD10对回归与二分类的保护强于全模型短续训，但没有分类收益；不能把它解释成现有表示已经达到理论上限。D1继续作为分类优先主点，R1继续单列低MAE候选。不追加新训练。

## 独立列出的开发集选点

每轮另存val-best-Acc7和val-best-MAE；评估两个checkpoint的完整eta后，仅按开发集MAE最小、Acc7最高、固定名称/eta顺序选择。候选表与选择审计见`val_selection_audit.json`，测试不参与此附加选择。

| 实验 | val-selected checkpoint / epoch / eta | 测试Acc7 | 测试MAE | 测试Acc2non0 | 测试macro-F1non0 |
|---|---|---:|---:|---:|---:|
| DETACH | best_mae / 49 / .9 | 43.002915 | .757112 | 83.689024 | 83.339623 |
| HEAD10 | best_acc7 / 9 / .8 | 44.023324 | .736407 | 84.298780 | 83.764522 |
| FULL10 | best_mae / 4 / .8 | 43.586006 | .748907 | 84.146341 | 83.721308 |

HEAD10/FULL10从test-selected D1 epoch59初始化，整个项目方案也参考过测试结果。另存val checkpoint不消除这些来源的选择偏差，不能称为独立未触及测试集的确认，更不是配对多seed结论。

## 干预是否实际生效

- HEAD10四个保留checkpoint的所有非head_cls7张量与父权重逐位相同；两test-selected checkpoint在val/test共1830条输出中，原始回归值与父点最大差为0。冻结区eval和每轮覆盖自动解冻均生效。
- HEAD10/FULL10各410次优化器更新，分类头初始LR同为2.25e-5，常数LR、新optimizer；FULL10其他可训练组用D1初始LR。冻结臂dropout设置是干预的一部分，不把差异全部归因于可训练参数数目。
- DETACH为8200次更新。微型测试确认前向不变、邻居特征梯度切断、中心和gate参数梯度保留；实际执行启用了该钩子。
- 新逐样本输出各自重建28个val/test点，误差均≤1e-5。独立标准库检查器从匿名输出重算42个test-selected测试eta点。

## 邻域与分支诊断

统一eta=.6，按test-best-MAE checkpoint比较：D1的异极性邻居组MAE=.761806，DETACH为.760707；非冲突组D1=.707329、DETACH=.703469。回归改善较小，异极性邻居组正确档位数从144降至142，非冲突组从161降至155；没有证据支持“切断邻居梯度即可同时改善极性与强度”。当前测试集无邻居组样本为0，无法估计该组效果。

这些是不同训练轨迹选中checkpoint的开发比较，不是同权重的前向反事实；样本分组也不能单独证明梯度冲突成因。完整分组、gate/方差及实际特征变化见`study_results.json`中的`output_verification`。

## 过程梯度证据的边界

DETACH、FULL10的裁剪前全可训练参数范数分别均值24.27/24.92，记录中每步都超过clip=1；HEAD10均值1.00、45.1%步骤超过1。参数范围不同，范数不能直接用来解释精度高低，记录只说明裁剪实际频繁生效。

首个训练batch的router-phi子空间中，FULL10加权分类与回归梯度余弦约-.767，DETACH约-.314；二者起点不同（父checkpoint与预训练初始化），不能把两数之差作为detach缓解冲突的因果证据，也不能外推到所有batch或整个编码器。HEAD10共享router冻结，该项记为不适用，不伪造零梯度夹角。

原始裁剪前范数见`training_process/`，分项探针见`study_results.json`。纯标准库验证：`python tools/check_mosi_studies.py`。没有重跑旧CDF/EMD、尺度大网格或其他seed。
