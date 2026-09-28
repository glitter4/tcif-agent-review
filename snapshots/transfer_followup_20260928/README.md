# 迁移后续执行分支

独立Lab MOSI分支，不替换根目录MOSEI主线。基础模型/数据/损失依赖见[既有MOSI执行快照](../mosi_diagnostics_20260922/README.md)，本目录覆盖有改动的训练、初始化、评估与普通上下文模块。

- `server-code/transfer_support.py`：完整/选择性权重继承、52张量fresh重置、严格形状与张量核验、精确暴露sampler、计算量trace。
- `server-code/context_variant.py`：普通masked均值+残差MLP；源自已审查普通融合实现，针对MOSI context_aux=0按活跃posterior参数匹配宽度，prior投影单列。
- `server-code/grs_controls.py`：仅复用原S的extra_loss，未启用G/R；权重/边界/reduction不变。
- `server-code/train_emotion.py`：新增模式和trace、S、普通融合构造与metadata；目标仍test-selected双checkpoint。
- `server-code/eval_all_mosei_maefixed.py`：从metadata严格恢复普通融合，增加只读validation-only/domain override，不更改读出公式。
- `scripts/run_followup.py`：有上限两GPU队列、两新增source依赖、98点目标评估、2×2交互、router变化与配对修复。
- `scripts/test_followup.py`：重置边界、零残差可学习、mask及活跃容量、2044/65304预算、原S边界测试。
- `scripts/validation_comparison.py`、`reference_drift.py`、`repair_class_audit.py`：补齐对照诊断，不新增训练。

实际source/preflight字段见结果manifest；机器路径已替换占位，测试导入按本包布局调整。必须提供数据/cache/模型并按实际运行树目录布局配置脚本，不能clone即复现。原训练器的字典散列种子设置按仓库约束移除。训练小补丁见scripts/training_patch.diff。
