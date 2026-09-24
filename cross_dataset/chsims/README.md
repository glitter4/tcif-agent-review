# CH-SIMS 实际执行分支与诊断工具

这是2026-09-22 C1实际执行树的独立快照，不能与仓库根目录的MOSEI代码混用。训练及评估入口位于`source/server-code/`，模型参数/forward保持既有结构；pathstudy新增可选的邻居梯度阻断与短程训练模式。

## 先看这些实际函数

| 问题 | 文件与符号 |
|---|---|
| 中心L1是否重复叠加 | [train_emotion.py](source/server-code/train_emotion.py) `compute_center_regression_loss`、`loss_reg`与`loss_raw`；中心loss返回L1×weight，context仍独立SmoothL1 |
| 上下文/门控监督 | 同文件 `_compute_tcif_context_aux_loss`、`_compute_tcif_transition_gate_loss` |
| 五分类中心与分箱 | [head_utils.py](source/server-code/head_utils.py) `get_cls7_centers`、`continuous_to_cls7_hard`，CH-SIMS有专门分界，不能直接等比例套MOSI |
| expected/argmax融合 | 同文件 `compute_final_prediction`；先沿现有评估规则截断回归，再按eta融合 |
| 二分类和guarded选点 | [chsims_followup_support.py](source/server-code/chsims_followup_support.py) `binary_metrics`、`update_guarded`；零阈值，预测0归正类 |
| test双checkpoint | `train_emotion.py::_update_best_checkpoints`；评估器发现白名单见[eval_all_mosei_maefixed.py](source/server-code/eval_all_mosei_maefixed.py) |
| 邻居任务特征detach | [models_emotion.py](source/server-code/models/models_emotion.py) `_apply_tcif_to_local_output`；默认关闭，只阻断context reg/cls特征 |
| 仅分类头训练 | [pathstudy_support.py](source/server-code/pathstudy_support.py)；每epoch在常规解冻逻辑后再次固定requires_grad及eval模式，避免文本层重新解冻 |
| 邻居与文本输入 | [emotion_dataset.py](source/server-code/datasets/emotion_dataset.py) `CHSIMSProcessDataset`、`build_temporal_context_index` |

已有head_cls7是线性分类层，和TCIF context classification head不同；head-only只训练前者。原名为head的优化器组包括更多参数，不等同此处head-only。

## 配置与运行边界

[plan.json](plan.json)包含新6组配置，[reference_l1.json](reference_l1.json)是原B0配方。detach各50轮，head-only/full-continue各seed各5轮；不重训既有B0。后两者从相同B0 test-best-MAE出发、重建AdamW，分类头LR均6.96e-5，其余组原LR、恒定调度；它不是完整optimizer resume。

主结果始终采用test-selected checkpoint及test eta扫描。每个保存checkpoint分别跑expected/argmax七点eta；附加val checkpoint在既有gate/context记录中仅作旁证。

这是脱敏执行证据，不是一键可运行发布包。`/path/to/...`须由实际数据、缓存、backbone、输出根与Python环境替换；runner内的C1 hostname和调度参数是原执行环境约束。使用原环境的torch2.1.2+cu118及所需transformers/torchvision等。未附任何模型权重或音视频。

原文件中设置Python字典散列种子的语句按仓库规则移除；没有生成文件摘要。其余打包改变限于路径脱敏与包标记，来源和字节数见[source_inventory.json](source_inventory.json)。未包含未使用的magnitude adapter等历史分支。

## P0导出与核验

[export_predictions.py](export_predictions.py)在原评估器上注册forward hooks，每个checkpoint仅导出一次val/test；不为每个eta重复推理。它包含模型构建、strict state load和原dataset读取路径，保留样本ID与邻居信息。

[analyze_outputs.py](analyze_outputs.py)使用相同head_utils重建224个既有val/test结果，逐样本比较误差和分箱；完成1120点有限幅度扫描。a按val选择、eta仍按test选，不能因此称为独立测试。

[check_pathstudy.py](check_pathstudy.py)验证detach梯度边界和前向等价、仅head更新与回归不变；[verify_freeze.py](verify_freeze.py)在实际训练后逐参数比对，并检查eta0回归预测。

逐样本数据与完整诊断见[诊断目录](../../results/chsims_diagnostics_20260922/diagnostics.json)。

## 新训练结果已完成

[六组完整结果、冻结核验和GPU迁移说明](../../results/chsims_pathstudy_20260922/REPORT.md)：182点齐全，detach原始输出已附，四组短程对照附加原始输出待补。

CH-SIMS时序辅助监督归零对照已完成：[两seed四组完整结果](../../results/chsims_temporal0_20260923/REPORT.md)。C1提交时不可达，使用Lab5090同机B0进行配对。
