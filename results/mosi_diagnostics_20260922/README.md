# MOSI：过程诊断与受限校准

2026-09-22。本包新增D1与router-temperature R1的实际逐样本数值，不含文本、音视频或checkpoint。四个checkpoint各包含229条val、686条test，共3660条记录。D1两个checkpoint同为epoch59，仍分别导出；R1分别epoch58/149。

**结论：本轮受限幅度校准没有收益。** 每个checkpoint先仅按开发集MAE选择eta（平局看Acc7、再较小eta），再在a=.9/.95/1/1.05/1.1中筛MAE不劣于a=1的点，按Acc7、MAE、接近1排序。四次均选a=1；没有扩大参数范围。D1两checkpoint选eta=.8；R1 best-Acc7选.8，best-MAE选1。

父checkpoint本身按test选择，所以这不是独立留出测试。新增标定参数只由开发集决定，测试不参与挑eta或a。原七点test-selected结果独立保留。

## 数据与验证

- `samples/<D1或R1>/<checkpoint>/<val或test>.jsonl`：完整float32有效精度的回归原始/截断输出、7维logits、local/prior输出、邻居ID/mask/位置/标签、TCIF gate、两类方差均值、innovation norm及真实posterior-local特征相对变化。
- sample/group ID替换为可稳定连接的顺序ID；同split在D1/R1、不同checkpoint中一致。原ID映射留在实验工作区，不发布；无散列。
- `reconstruction_audit.json`：56个val/test重建点逐样本与历史CSV匹配，容差1e-5。
- `calibration_results.json`：开发集完整候选与冻结选择、测试同点结果。
- `error_decomposition.json`：分支互补、近零、同符号错档、跨符号、邻域极性及视频组统计。
- `*_epoch_records.txt`及`*_log_coverage.json`：提取原有日志，不伪造历史上未记录的梯度范数/夹角。

纯标准库复核：`python tools/check_mosi_diagnostics.py`，可重建28个测试eta点并验证样本连接和符号不变性。它独立于Torch重算读出，不重新运行模型。

## 有证据的观察

| R1 checkpoint | 回归MAE | 分类期望MAE | eta=.6融合MAE | 两路残差异号样本 |
|---|---:|---:|---:|---:|
| epoch58 best-Acc7 | .748825 | .748141 | .740124 | 77/686 |
| epoch149 best-MAE | .753505 | .737816 | .730642 | 106/686 |

后期回归分支没有改善，分类分支和误差互补发生变化；这不是训练梯度冲突的证明。R1后期local输出在eta=.6的MAE=.801492、posterior为.730642，说明其上下文前向修正明显；这种同模型内部诊断也不是移除TCIF并重训的因果对照。

D1 eta=.6：66条弱非零样本中有32条跨符号错误；217条强情感样本中有104条同符号错档。测试集有327条带至少一个异极性有效邻居，359条无异极性有效邻居，**无有效邻居样本为0**，因此无法在当前测试集估计无邻居组效果。各错误类别可能与其他分类统计重叠，不相加伪造错误总数。

## 已去重与后续训练状态

旧MOSI wave38/39已经分别测试softCE+flat-logit CDF/EMD .5、hardCE+CDF/EMD .2，均退化，见[历史摘录](../evidence/mosi_ordinal_history.md)。旧scale/bias/集成校准与本次受限单模型诊断区分，不重跑旧网格。

新训练已经启动：DETACH（D1配方200轮，邻居特征detach）、HEAD10（仅head_cls7精调10轮）、FULL10（相同步数全模型继续训练）。两短臂从D1 epoch59初始化、分类头LR2.25e-5、常数LR、新optimizer；其他设置固定。HEAD10冻结区eval，并每轮覆盖原自动解冻逻辑。当前没有三项最终成绩，状态不写成完成。原test-selected双checkpoint保留，另外单列val-selected双checkpoint及其完整eta扫描；初始化来源的test-selected局限继续披露。

微型测试已验证detach前向逐位相同、邻居特征梯度被切断、中心与gate梯度保留；head-only参数冻结和回归不变、全模型控制参数组无遗漏。实际训练补记裁剪前梯度范数及首个训练batch的router-phi子空间分项梯度；子空间结果不能推广成整个共享编码器的梯度结论。

实际执行分支及干预代码见[源码范围说明](../../snapshots/mosi_diagnostics_20260922/README.md)。
