# 去重与暂缓记录

当前范围CH-SIMS，MOSI已有记录保留但不在本轮重训。

| 项目 | 当前处理 |
|---|---|
| gate监督tau=.4/.25、context auxiliary=.03 | 已完成两seed八组，直接复用；context在seed41改善、seed40退步 |
| 中心L1及其.75/1/1.25权重 | 已实现并试验，不重复增加中心L1损失 |
| seed41–44常规复验 | 已完成，无联合突破，不继续无目的换seed |
| 符号BCE .005/.01 | 已完成且没有联合改进，不重复 |
| legacy_sign_safe | 曾在SmoothL1基线完成，无更好均衡结果；不等同于已证伪L1上的效果，但本轮不优先重复 |
| 英文文本列和模型类型 | 已核实；本次补了真实token及截断统计，没有改输入 |
| 逐样本融合CSV/gate记录 | 已存在，复用；本次补raw回归、5维logits、local/posterior和相对残差 |
| 共同正比例幅度校准 | 本次五档已完成；无新增联合合格点，不扩大密集网格 |
| 邻居任务特征detach | 本次新两seed对照；不同于context降权及历史router stop-gradient |
| 冻结表示，仅head_cls7短程训练 | 本次新两seed对照，配相同起点和步数的full-continue；不重复B0训练 |
| flat logits CDF/EMD | 暂缓；历史ordinal head/其他数据集EMD不能当作当前CH-SIMS方案已完成 |
| 弱非零/精确零有限hinge | 暂缓；不同于BCE，但先等梯度路径与分类头结果，不直接套MOSEI边界参数 |

历史test-selected开发口径保留。eta继续在test选择；本次幅度a按val选择的结果明确单列，不将整个流程包装成独立测试。

新结论与实际完成状态见[实验总览](../../results/EXPERIMENTS.md)，[P0诊断](../../results/evidence/chsims_diagnostics_report.md)。
