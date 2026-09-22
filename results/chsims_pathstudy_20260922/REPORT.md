# CH-SIMS 邻居梯度与分类头对照结果（2026-09-22）

**六组训练与全部182个eta评估点已完成；两组detach的原始输出已附，四组短程对照的附加原始输出仍待补。** 本报告不将数据导出待补写成指标未完成。

## 本轮结果与选点规则

主口径为**测试集checkpoint与测试集eta选择**。每个实际保存checkpoint分别完成expected/argmax的eta={0,.2,.4,.6,.8,.9,1}。下表展示argmax网格内Acc5最高、同分MAE低的代表点，各行恰为eta=.8；不是固定eta评估。四项指标及Acc2(all)均来自同行同一checkpoint/读出/eta，F1为非零macro-F1，百分比单位%。

| Run | checkpoint | epoch | eta | Acc5↑ | MAE↓ | Acc2non0↑ | F1non0↑ | Acc2(all)↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DETACH_S40 | best_acc7_model | 47 | 0.8 | 46.389 | 0.415238 | 80.670 | 79.437 | 76.586 |
| HEAD_ONLY_S40 | best_mae_model | 3 | 0.8 | 47.265 | 0.403465 | 82.732 | 81.477 | 76.149 |
| FULL_CONTINUE_S40 | best_acc7_model | 5 | 0.8 | 46.171 | 0.412184 | 81.443 | 79.947 | 76.368 |
| DETACH_S41 | best_mae_model | 23 | 0.8 | 50.109 | 0.390312 | 82.990 | 81.976 | 77.899 |
| HEAD_ONLY_S41 | best_acc7_model | 2 | 0.8 | 46.171 | 0.402019 | 81.443 | 79.885 | 74.179 |
| FULL_CONTINUE_S41 | best_mae_model | 3 | 0.8 | 45.952 | 0.415072 | 79.124 | 77.732 | 73.523 |

HEAD_ONLY/FULL_CONTINUE的epoch是新增5轮内的轮次；DETACH为从头50轮。所有新训练采用seed40/41配对，不把不同seed的最佳指标拼接。

## 结论

1. **DETACH_S41是新增Acc5/MAE取舍候选**：test-best-MAE epoch23、argmax eta=.8为50.109409/.390311629/82.989691/81.976351，Acc2(all)=77.899344%。它通过原guard护栏；原50/.390/82/81联合目标仍因MAE超出约.000312而未达成，新84/49/.402/82目标也未达成。best-guarded与best-MAE同为epoch23，仍各自保存并完整扫描，不能删去重复epoch记录冒充少评一次。
2. detach的seed40退步，**不支持跨seed稳定提升**。相对历史dl01 CH_L1，seed41的Acc5、MAE、F1和Acc2(all)更好，Acc2non0略低；保留为取舍候选，不能宣称所有指标全面优于历史点。C003_S41的.389766仍更低，但分类表现弱于新detach点。
3. **仅分类头精调优于同seed全模型继续训练的代表点**。seed40相对起点只有很小的MAE/F1变化、二分类略降；seed41相对自身B0 best-MAE起点四项改善，但仍不及历史最佳候选。不能仅比较两个续训方法而不检查epoch0。
4. 不重复既有B0训练；不新增CDF/EMD或符号hinge。已有小尺度校准未带来联合突破，见[P0报告](../evidence/chsims_diagnostics_report.md)。

## 对照设置与冻结核验

DETACH只切断输入TCIF的邻居reg/cls任务特征梯度，前向与参数结构不变；tau=.75、context aux=.1、中心L1×1等B0配方保留。中心与TCIF模块仍学习。

两种续训都从各seed既有B0 test-best-MAE出发，5轮、fresh AdamW、恒定LR，分类头LR=6.96e-5；FULL_CONTINUE其余原可训练参数组沿用原LR，不解冻原本冻结的整个backbone。HEAD_ONLY只更新head_cls7并固定其余模块的dropout/eval状态。两臂同起点、初始seed和步数；不同随机数消耗不保证后续epoch逐batch顺序完全相同。

实际逐参数核验显示，两组HEAD_ONLY的所有已保存checkpoint均仅head_cls7.weight/bias变化；val/test eta0回归预测最大差**0**。证据：[seed40](freeze_verification_s40.json)、[seed41](freeze_verification_s41.json)。

## GPU迁移与数值核验

C1节点因GPU故障DRAIN，detach训练50轮已完成，后续评估曾落到CPU。传输遇连接重置后改为16MiB分块断点续传，保留已传数据；将5个已选checkpoint迁到独立Lab5090工作树，未重训。seed41最后一个导出曾触及文件数上限，提高NOFILE到65536后仅补该checkpoint，最终10个val/test导出完整。

主指标由4组C1评估和2组Lab5090 CUDA输出重建组成。Lab为torch2.7.1+cu128/transformers4.45.2，与C1环境不同；标签NPZ与英文CSV逐字节一致、缓存ID覆盖完整。18个已有CPU测试预测文件与GPU交叉比较：最大绝对差4.1425228118896484e-6，二分类符号变化0，见[比较记录](gpu_cpu_recovery_comparison.json)。这不是对全部缓存字节或全部算子的等价证明。

原CPU部分输出保留，原两项CPU评估被GPU恢复替代，不能描述为原作业正常退出0。汇总已完成；四项额外原始输出仍在C1显示JobHeldAdmin，明确待补。

## 数据与实现

- [完整六组配置、182点结果](summary.json)
- [产物完成/待补状态](artifact_status.json)
- `predictions/DETACH_S40/`和`predictions/DETACH_S41/`：5个checkpoint×val/test，共10个JSONL、4565条原始回归/logits及TCIF记录；含guarded checkpoint。
- [原B0/C003对照网格](../chsims_gatectx_full_sweeps.json)、[P0原始输出与诊断](../chsims_diagnostics_20260922/diagnostics.json)
- [CH-SIMS实际执行实现与协议](../../cross_dataset/chsims/README.md)、[去重清单](../../cross_dataset/chsims/DEDUP.md)

本次只同步已完成结果与可用证据，未提交额外训练。
