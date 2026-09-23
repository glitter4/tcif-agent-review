# MOSI配对梯度探针源码

该快照是与根目录MOSEI实现分开的MOSI实际运行变体。基础模型、损失、数据和评估实现见[前一份MOSI执行快照](../mosi_diagnostics_20260922/README.md)；本目录只提供新增探针入口和修改过的训练器/控制文件，不覆盖原源码。

- `server-code/train_emotion.py`：在原损失组装之后、`loss.backward()`与optimizer step之前拦截前8个microbatch。诊断结束直接退出，不进入更新和epoch末选模。
- `server-code/study_controls.py`：载入既有权重，诊断模式下可给两个TCIF邻居特征添加detach钩子。
- `server-code/study_gradprobe.py`：按两个microbatch累积加权L1、softCE与gate分项梯度，记录各参数组连接、范数、内积、余弦和实际全局范数；结束比较所有权重/buffer。
- `scripts/probe_runner.py`：D1/R1共12个固定条件的执行清单。
- `scripts/retry_late.py`：首次GPU显存竞争后只补R1后期缺失三项，不重复成功探针。
- `scripts/neighbor_coverage.py`：用实际训练标签和邻域构造函数核查抽样覆盖；仅输出聚合计数。

真实机器路径替换为`/path/to/user`，无数据、模型权重或原始训练样本ID。脚本需外部训练环境、dataset/cache和原checkpoint才能运行；不能从此快照直接复现论文分数。
