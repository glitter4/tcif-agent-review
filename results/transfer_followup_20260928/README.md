# MOSI迁移后续：四组实验与计算量对照完成

2026-09-29核查。7个新MOSI目标阶段均完整200轮，两个新增source阶段均固定4逻辑轮，全部exit_code0。新目标共98点，另复用原C-ST完整14点；每个目标checkpoint单独完整eta={0,.2,.4,.6,.8,.9,1}、expected/T=1。没有增加seed、组合或重训。

**未出现最新46.1/85/85/.730联合达标点。主要结论：降router LR确实减小参数漂移，但不等于性能胜出；S有依赖checkpoint的取舍；整块TCIF重置没有全面收益。源任务迁移优于本次计算量匹配的额外MOSI重复训练；2×2不支持单seed下“TCIF特别适合迁移”的笼统结论。**

## 1—3：干预结果

以下每行来自同checkpoint、同eta。A*=max(Acc2,Acc2non0)，F*=max两列macro-F1，本表均为nonzero来源；weighted另存，不参与max。

| 实验/用途 | checkpoint / epoch | eta | Acc7 | MAE | A* | F* |
|---|---|---:|---:|---:|---:|---:|
| 原完整迁移均衡点C-ST | best_acc7 / 58 | .8 | 46.209913 | .709394 | 84.146341 | 83.769175 |
| E1降router，最高Acc7 | best_acc7 / 71 | 1 | 46.064140 | .717794 | 85.060976 | 84.561444 |
| E1最低MAE | best_mae / 58 | .9 | 45.481050 | .711946 | 83.689024 | 83.354479 |
| E2迁移+S，最高Acc7 | best_acc7 / 55 | .9 | 46.064140 | .727445 | 83.689024 | 83.308880 |
| E2较均衡取舍点 | best_mae / 58 | .8 | 45.481050 | .707711 | 84.756098 | 84.331410 |
| E2最低MAE | best_mae / 58 | .6 | 45.189504 | .707401 | 84.451220 | 84.034359 |
| E3重置TCIF，最高Acc7 | best_acc7 / 78 | .6 | 46.647230 | .718113 | 83.689024 | 83.324424 |
| E3最低MAE | best_mae / 93 | .8 | 45.189504 | .711663 | 84.603659 | 84.229831 |

E1的85.060976超过A*=85，但Acc7仍不足46.1、F*仍不足85，不能算联合达标。E3最高Acc7与原完整迁移最高值相同，但同eta=.6的MAE从.710216升至.718113；整块重置不应成为默认新基线，也不能从中单独归因gate或方差估计。

### E1：参数变化与两域验证性能

best-Acc7的两层phi相对源权重L2变化：原迁移为.39148/.46397，E1为.13625/.18530；best-MAE原迁移为.31692/.40177，E1为.12608/.16948。降低LR确实保留了更多原参数数值，但这些是各自选中epoch的比较，不是相同步数参数轨迹；raw phi变化也不是专家分配变化的完整描述。scale单独保留在router_drift.json中。

固定expected/T1、eta=.8，只作附加验证诊断，不用于选点：

| 权重 | MOSEI val Acc7 / MAE | MOSI val Acc7 / MAE |
|---|---|---|
| source epoch4 | 57.1352 / .470976 | 34.0611 / .880881 |
| 原迁移best-Acc7 | 47.7819 / .623005 | 48.9083 / .736210 |
| 原迁移best-MAE | 50.2940 / .586904 | 45.4148 / .728966 |
| E1 best-Acc7 | 48.9043 / .622844 | 47.5983 / .743009 |
| E1 best-MAE | 46.0716 / .634161 | 48.0349 / .738590 |

源性能下降伴随目标适配，不自动等同有害遗忘；E1漂移更小也未一致保住源MAE或提高目标四指标。完整数据：[validation_comparison.json](validation_comparison.json)、[原迁移router漂移](reference_router_drift.json)、[E1漂移](runs/E1_ROUTER/router_drift.json)。

### E2：弱正/弱负修复与新增错误

统一best-MAE类型、eta=.8，对同一批测试样本与原迁移best-MAE比较：

| 群体 | n | 修复符号 | 新增符号错误 | 修复Acc7 | 新增Acc7错误 |
|---|---:|---:|---:|---:|---:|
| 弱正 | 37 | 3 | 0 | 2 | 1 |
| 弱负 | 29 | 1 | 1 | 2 | 1 |
| 中强非零 | 590 | 7 | 6 | 31 | 26 |
| 精确零 | 30 | 4 | 0 | 1 | 3 |

弱正3次符号修复中：1条原本Acc7正确并保持正确，1条变为Acc7正确，1条从Acc7正确变错；弱负1次修复保持Acc7正确。排除零标签后的符号净修复4条、全样本Acc7净修复5条。相同读出下MAE由.707052微升至.707711，属于小幅取舍；在best-Acc7类型eta=.8，弱负修复2条但弱正新增2错，中强群体净增4错，不能泛化成S总能改善弱极性。精确零不在S监督mask内，零样本变化属于间接影响。

[完整配对修复](runs/E2_S/paired_repairs.json)、[符号修复的Acc7状态转移](runs/E2_S/paired_repairs_detailed.json)覆盖val/test、两checkpoint、全部eta，不只给净提升。

### E3：选择性初始化核验

52个TCIF/context-head张量恢复同seed初始值，743个其他张量与源权重逐值相等；两个残差输出投影weight/bias为0、shape不变、新optimizer。实际核验见[初始化记录](runs/E3_RESET/initialization_audit.json)。这项定位只针对整块TCIF，不能单独推断gate或统计参数的作用。

## 4：严格同口径2×2

新C-T/B-T均完整200轮，C-ST复用已完成源4+目标200，B-ST新跑普通融合源4+目标200。旧Lab D1中断结果不放入四格。

普通融合为masked邻居均值+拼接残差MLP，无概率filter/gate及gate loss；context aux仍为0。每分支活跃posterior MLP参数1526948，对TCIF参考1525666，差.08403%；诊断用prior投影另计，因此全模型总参数普通394834589、TCIF394229721，不伪称总参数完全相同。

**统一best-Acc7 checkpoint、eta=.8：**

| 四格 | Acc7 | MAE | A* | F* |
|---|---:|---:|---:|---:|
| B-T 普通融合，仅MOSI | 43.586006 | .742974 | 82.774390 | 82.338997 |
| C-T TCIF，仅MOSI | 44.460641 | .763996 | 82.774390 | 82.321475 |
| B-ST 普通融合，MOSEI→MOSI | 45.626822 | .740062 | 82.469512 | 82.060946 |
| C-ST TCIF，MOSEI→MOSI | 46.209913 | .709394 | 84.146341 | 83.769175 |

这一口径下C-ST优于B-ST的四指标。但交互项`(C-ST−B-ST)−(C-T−B-T)`为：Acc7 **−.291545**、Q=−MAE **+.051690**、A* **+1.676829**、F* **+1.725751**。分类交互并非正向。

**统一best-MAE checkpoint、eta=.8：**

| 四格 | Acc7 | MAE | A* | F* |
|---|---:|---:|---:|---:|
| B-T | 42.857143 | .740263 | 84.146341 | 83.737605 |
| C-T | 42.711370 | .737623 | 83.689024 | 83.243214 |
| B-ST | 45.189504 | .703300 | 84.451220 | 84.001377 |
| C-ST | 44.752187 | .707052 | 84.146341 | 83.634619 |

这一口径下普通融合B-ST四项反而更好；交互项为Acc7 −.291545、Q=−MAE −.006393、A* +.152439、F* +.127633。因此不能笼统写“TCIF在迁移下始终有增益”，更不能写“TCIF特别适合迁移”。这是单seed、test-selected对照，不是稳定多seed交互证据。其他eta及两checkpoint类型的完整四格见[factorial.json](factorial.json)。

B-ST最低MAE为best-MAE epoch58、eta=.6：**44.314869/.701917/84.756098/84.331410**，是有价值的低MAE候选，但仍无联合达标。

## 额外MOSI计算量对照

COMPUTE_SOURCE每逻辑epoch循环使用MOSI train的16326条暴露，4轮精确2044更新/65304暴露，随后新optimizer正常MOSI200轮。B_SOURCE也实测2044/65304，二者记录的输入张量元素总量逐字段相等。源训练墙钟分别.3514小时（额外MOSI）和.4068小时（普通融合MOSEI），受模型和资源状态影响，不能将墙钟或输入规模当成精确FLOPs相等证明。

COMPUTE_T的最高Acc7同点：best-Acc7 epoch11、eta=.9，44.314869/.769249/83.536585/83.191459；最低MAE同点：best-MAE epoch58、eta1，42.857143/.751828/83.689024/83.190243。固定best-Acc7/eta=.8为43.148688/.772295/83.536585/83.206281，明显弱于C-ST同口径；本seed支持源监督提供了额外价值，而不只是多做2044次MOSI更新，尚不代表所有预算或seed下的普遍结论。

## 产物及检查

- `runs/<ID>/result.json`：7项共98点，所有原始all/nonzero/weighted字段及A*/F*来源。
- [C-ST复用证据](C_ST_reused.json)、[总完成记录](complete.json)、[配置清单](manifest.json)。
- `runs/<ID>/compute_trace.jsonl`：每目标8200更新/256800暴露；`source_runs/`保存两源4轮、2044/65304及固定权重元数据。
- `initialization_audit.json`、`router_drift.json`、`paired_repairs.json`及源val附加诊断保留。
- [执行源码与测试](../../snapshots/transfer_followup_20260928/README.md)。

运行`python tools/check_transfer_followup.py`与`python tools/check_bundle.py`。方差/innovation对普通融合不可用，记N/A，不当成预测NaN；发布数值如有非有限诊断转null并记录availability.json，核心指标须有限。没有上传权重、cache、原始数据、真实样本ID或私有路径。三项迁移干预和B-ST使用额外MOSEI监督，不能冒充与MOSI-only完全相同数据设置。
