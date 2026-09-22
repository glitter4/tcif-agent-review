# MOSI 独立执行快照

该目录补充实际Lab5090 MOSI执行代码，**不替换仓库根server-code的MOSEI快照**。

- `server-code/train_emotion_baseline.py`：D1/R1原训练代码。
- `server-code/train_emotion.py`：本次受控训练版本，增加study_controls调用、独立val checkpoint记录、首batch分项梯度与逐步裁剪前梯度记录；原模型参数形状不变。
- `server-code/study_controls.py`：邻居特征detach钩子、head-only冻结、加载固定父权重、匹配的分类头LR、过程诊断。
- `server-code/models/models_emotion.py::_apply_tcif_to_local_output`：中心/邻居任务特征、两套TCIF与最终回归/分类读出。
- `server-code/models/tcif.py`：local/prior/posterior、gate与实际特征修正。
- `server-code/head_utils.py`：实际类别中心、分箱、soft targets、expected读出。
- `server-code/train_emotion.py::_compute_tcif_transition_gate_loss`与`loss_raw`：gate监督和总损失；`_update_best_checkpoints`：主test-selected及另存val对照。
- `server-code/eval_all_mosei_maefixed.py`：checkpoint重建、截断及实际读出。
- `scripts/export_outputs.py`：无梯度forward hooks导出；不改变模型输出。
- `scripts/analyze.py`：原ID下的逐点重建、开发集校准与分组分析；发布的数据已另行做顺序ID替换。
- `scripts/test_interventions.py`：受控干预的梯度/冻结测试。
- `scripts/train_study.py`：三项固定预算训练及独立的test/val完整eta评估。

机器路径改为`/path/to/user`，移除字典散列种子环境设置，遵循本仓库打包约束。示例执行脚本需要配置真实运行位置、外部数据/模型/cache与checkpoint，不能直接clone后训练。源码快照不能代替缺失权重或完整硬件环境。

本批覆盖MOSI，不意味着CH-SIMS center_reg扩展已被核实或纳入；CH-SIMS仍按其独立证据解释。原有上游归属见根目录ATTRIBUTION.md。
