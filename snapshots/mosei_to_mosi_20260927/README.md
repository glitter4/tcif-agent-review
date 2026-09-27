# 同构监督迁移执行增量

基础为Lab D1的MOSI同形训练实现，未改模型结构。模型/数据/读出依赖可参照[既有MOSI执行快照](../mosi_diagnostics_20260922/README.md)；本目录训练器不含HEAD10/DETACH或GRS干预。

- `server-code/transfer_support.py`：固定源4轮约束、目标键/shape/数值严格加载、新optimizer、源val记录。
- `server-code/train_emotion.py`：明确source/target阶段；source模式test dataset引用val作为未使用占位且关闭test评估，跳过best-checkpoint选择；源仅保存latest固定最终权重。
- `scripts/run_transfer.py`：预检、一个源阶段、一个MOSI目标阶段、双checkpoint完整eta及目标结果来源列。
- `scripts/training_patch.diff`：相对D1原训练器的小补丁。
- `scripts/test_transfer.py`：严格加载相等、不兼容shape拒绝、源不选择best或读取test指标的测试。
- `scripts/audit_data.py`：视频ID交集及cache覆盖检查，不执行模型推理。

路径改为占位；原训练器的字典散列种子设置按仓库约束移除。依赖真实dataset/cache/backbone与checkpoint，本目录不能直接clone后运行。测试导入路径为此包布局调整，测试逻辑未变。
