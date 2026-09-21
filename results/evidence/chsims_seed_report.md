> 历史记录摘录；数字保留原精度，省略机器位置和调度记录。来源文件名见 docs/PROVENANCE.md。

# CH-SIMS L1×1 seed41–44结果

四组均完成50轮和完整评估。seed43首次348756_2因其他进程占用GPU显存而OOM；重试348764已COMPLETED/退出0。seed41/42/44及汇总348760同样COMPLETED/退出0，无缺失结果。每组两种checkpoint×双读出×七点eta=28点，共112点，逐checkpoint核对eta集合完整。各组逐轮记录均50行。

## 结论

没有出现新的联合达标点，继续保留此前dl01 seed40 CH_L1推荐点。四组guarded eligible_epoch_count均为0；所有112点中最高Acc5仅47.484%，均达不到49%护栏，因此也没有达到原Acc5≥50%的联合目标。不能将“无第三checkpoint”误判成评估遗漏。

以下统一展示test-best-Acc5 checkpoint（文件名best_acc7_model）、argmax eta=.8、T=1。每行同一checkpoint同一eta；Acc/F1单位%，F1为非零macro。best-Acc5 checkpoint在训练中按expected eta1保存，不代表它在argmax eta=.8必然优于另一个checkpoint。

| seed | epoch | Acc2non0 | Acc5 | MAE | F1non0 | Acc2(all) |
|---|---:|---:|---:|---:|---:|---:|
| 41 | 31 | 81.701 | 47.265 | .397890 | 80.070 | 74.836 |
| 42 | 43 | 80.928 | 45.952 | .412877 | 79.261 | 74.617 |
| 43 | 31 | 80.670 | 45.077 | .407937 | 79.079 | 74.836 |
| 44 | 37 | 79.124 | 45.733 | .419996 | 77.670 | 72.648 |

seed41是新四组中最有价值的候选：上表点MAE比此前dl01推荐点的.399445略低，但Acc2non0、Acc5、F1和全样本Acc2都更低，不能替换均衡推荐。它在eta=.2时Acc2non0最高达到83.247%、F1non0=81.695%，但Acc5=39.387%、MAE=.411580，仍不满足联合目标。

此前C1 seed40 L1×1代表点为best-MAE epoch33、argmax eta=.8：Acc2non0=82.990%、Acc5=47.265%、MAE=.403593、F1non0=81.382%。此前dl01 seed40推荐点为guarded epoch26、argmax eta=.8：83.505%/49.672%/.399445/81.629%。上述参考采用的checkpoint与表格不同，且dl01与C1环境不同，不应将跨机差异归因为seed效应；本轮结果不能视作最佳seed的总体均值。

本次仅核查与归档，未提交新实验。继续搜索前建议先审计C1基线与历史结果的复现差异，当前四个新seed不支持“换seed已解决联合目标”的结论。
