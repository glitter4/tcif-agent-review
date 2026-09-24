# 打包验证记录

验证日期：2026-09-22。验证对象是整理后的上传目录，而非仅检查原服务器文件。

| 检查 | 结果 |
|---|---|
| Python语法、JSON解析、Markdown文件链接、本地导入依赖 | 通过，22个Python文件 |
| MOSEI消融结果完整性与CSV/JSON逐值一致 | 通过，70点 |
| MOSI D1/N1/N2/S1/R1双checkpoint七点eta | 通过，70点 |
| CH-SIMS四seed×双读出×双checkpoint×七点eta | 通过，112点 |
| 原有消融协议测试 | 3项通过 |
| 打包源码的小模型/消融/掩码/分类头测试 | 26项运行，25项通过，1项跳过 |
| 五种消融的计划命令生成 | 通过，未启动训练 |
| 模型/数据集/评估器/测试与来源的AST比较 | 除机器路径字面值外一致 |
| 训练器与来源比较 | 仅文档列明的路径、输出根和种子环境设置调整 |

本地标准库检查命令：

```bash
python tools/check_bundle.py
python -m unittest discover -s scripts -p 'test_tcif_paper_ablation.py' -v
python scripts/tcif_paper_ablation.py plan
```

模型测试在dl01独立临时副本中使用原执行环境、CPU和单线程运行：

```bash
cd server-code
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest test_tcif_ablation test_structv6_masking test_signed_reg_cls7_head -v
```

跳过项：`test_archived_full_filter_compatibility`，需要设置 `TCIF_REFERENCE_MODULE` 指向未打包的旧滤波器参考文件。本次没有以当前文件代替旧参考来制造兼容性结论。

模型测试覆盖五种变体的前向/反向、初始化、空邻域与padding、参数共享、保存重建、checkpoint元数据、标签映射和文本解冻时禁止cache等。测试使用合成小模型，不等于真实数据完整训练复现；本次没有训练、重新推理历史checkpoint或补跑敏感性eta网格。

结果证据保留原来源的数值和完成状态。对汇总结果的检查证明网格及导出完整，不能代替对未随包提供的逐样本预测、数据及权重的独立审计。

2026-09-22增量仅更新结果与文档及标准库检查器。重新验证新增28点与源结果逐值一致、两项配置各自仅一个字段变化、完成状态/轮次、router scale/temperature数值、Markdown引用及全部252点完整性。未改变模型代码，未重复执行上述旧模型测试。

随后新增的MOSI诊断快照另作验证：`python tools/check_mosi_diagnostics.py`通过3660行数值/匿名ID连接、28个测试eta点独立重建、开发集eta选择及符号不变检查；远端另核验56个val/test逐样本重建点与原CSV误差≤1e-5。干预微型测试通过detach前向相同、邻居梯度切断、中心/gate梯度保留、冻结参数范围和回归不变、控制臂参数组完整性。三项真实训练现已全部正常结束；`python tools/check_mosi_studies.py`进一步验证84个test/val-selected测试eta点、5490条新增输出、HEAD10原始回归不变、仅开发集选择和全部优化器步数。实际训练完成不等于精度改善，三项未联合达标。
## 最新CH-SIMS结果同步检查（2026-09-22）

新增8组448点逐checkpoint/读出/eta完整性检查通过，其中224点test-selected主结果、224点附加val结果。与工作区原始汇总直接比较，所有points对象逐值一致；仅机器路径脱敏与来源说明新增。合并同期MOSI增量后，全包700个完整历史/最新评估点检查通过，JSON解析、Python语法、Markdown链接和git diff空白检查通过。本次仅结果与检查脚本更新，未修改或重测模型代码。

## CH-SIMS诊断数据与训练路径检查

`python tools/check_chsims_diagnostics.py`验证16个JSONL共7304条、224点原预测重建、1120点校准及二分类/F1不变，固定val尺度选择规则与4×50轮过程记录完整。C1上check_pathstudy.py已通过detach前向等价/梯度边界和head-only参数/回归不变测试。模型参数的实际训练后检查由verify_freeze.py执行，运行状态另记，不能将小模型测试代替实际训练结果。

## CH-SIMS六组结果检查

`python tools/check_chsims_pathstudy.py`验证六组182点的checkpoint/readout/eta覆盖、10个原始输出的4565条记录、冻结核验与CPU/GPU对齐证据。指标汇总与本地原始summary直接比较，所有points逐值一致；附加原始输出待补状态单列。

## MOSI梯度探针增补

`python tools/check_mosi_gradprobe.py`仅用标准库检查12个无参数更新探针、48个有效batch、同一128条训练样本、D1三组完整/detach配对的回归及logits逐值相等、参数组互不重叠、裁剪系数与分项余弦数值，以及有效邻居覆盖。源12份结果与发布包直接逐值对比（仅原ID及私有路径脱敏）。不重复模型训练或checkpoint评估；已有双checkpoint完整eta协议未改变。

## MOSI G/R/S验收

`python tools/check_mosi_grs.py`通过3×200轮、42点评估、24600个有效更新记录、A*/macro-F*来源、G投影系数/条件和R/S损失权重及空mask检查。原始结果points与包内逐值一致；checkpoint epoch/GRS模式核对通过。C1实际环境已运行机制测试，验证累积后投影、非phi/辅助项保持、读出损失梯度和有限符号裕量边界。`check_bundle.py`通过新增源码语法和链接检查；本批未重复旧模型测试或训练。
