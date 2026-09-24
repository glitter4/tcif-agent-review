# MOSI G/R/S 执行增量

本目录为C1 D1恢复执行树上的独立训练逻辑增量，不替换根目录MOSEI实现，也不改模型参数结构。

- `server-code/grs_controls.py`：G条件投影、R读出L1、S弱非零有限裕量；默认baseline不启用新目标。
- `server-code/train_emotion.py`：实际训练器；显式GRS选项、checkpoint元数据、新loss及有效batch边界投影。
- `scripts/training_patch.diff`：相对C1 D1原训练器的完整小补丁，方便审阅介入位置。
- `scripts/test_grs.py`：累积后投影与microbatch区别、辅助/nonphi梯度保持、clamp/raw-L1、weak mask/正负平衡/达到裕量停止的机制测试。
- `scripts/run_grs.py`：三项配置、A*/F*定义、完整eta评估与阶段/最终目标逻辑。路径为脱敏占位，实际执行环境需自行配置。

模型、数据、标签读出等未改依赖可参照[既有MOSI源码快照](../mosi_diagnostics_20260922/README.md)；本训练器不包含该快照中HEAD10/DETACH干预。GRS不需要加载父checkpoint，三项从预训练backbone初始化重新训练。

机器路径已脱敏，字典散列种子设置按仓库约束移除。打包测试脚本的导入路径调整为本目录server-code，测试逻辑与C1一致。
