# CH-SIMS：从输出和训练过程拆分误差（2026-09-22）

本批P0已完成，无重复训练B0/C003：4组×2个test-selected checkpoint×val/test共16个JSONL、7304条样本记录。完整数据在[诊断目录](../chsims_diagnostics_20260922/diagnostics.json)，实际CH-SIMS执行实现见[独立分支说明](../../cross_dataset/chsims/README.md)。后续6组梯度路径/短程对照的状态单列，不将提交当作完成。

## 读出核验与幅度校准

原始回归、5维logits使用实际head_utils重建224个历史checkpoint/readout/eta/split结果。最大逐样本差2.384185791015625e-7，分类分箱和二分类符号差异均0。该224包含val/test；其中test原参考为112点。

共同幅度a={.9,.95,1,1.05,1.1}、无偏置、截断[-1,1]，共1120点；逐样本验证符号不变，所有Acc2/F1口径保持。每个checkpoint/readout/eta的a在val上优先Acc5且不增加val MAE；最终checkpoint/eta仍在test选择。这仍是test-selected开发结果，不能称为独立测试。

C003_S41 best-MAE/eta=.8的val所选a仍为1，保留MAE=.389766；eta=.9选a=.95后test MAE从.390399升至.393653，Acc5保持48.7965%。整个val选a网格没有联合合格点，最高Acc5仍48.7965%。当前不扩大尺度网格或加正负独立参数。

## 零标签、两头与上下文

C003_S41 best-MAE/argmax eta=.8：

- 69条精确零标签中，28条二分类计对，但只有23条Acc5计对；这两项不能混为同一错误。
- 388条非零样本中320条二分类计对，非零Acc5=51.289%。共同正比例校准无法修复全样本Acc2=76.149%不足76.3%的护栏。
- 135条同符号错档，109条跨符号错误，75条错至少两档；分组有重叠，不相加。
- 纯回归MAE=.436269，纯分类expected MAE=.410309，argmax eta=.8融合MAE=.389766。好融合不能解释为回归头变好。
- local→posterior的argmax eta=.8 MAE约.424129→.389766，218样本误差降低、213升高，其余相同或极小变化。不能仅由跨样本平均推断所有邻居有益。
- reg特征平均相对残差=.001940，cls=.047544；gate均值约.541/.685。gate值与最终特征/分数修正不是同一个量。

[分组上下文变化](../chsims_diagnostics_20260922/context_effect_audit.json)包含每个checkpoint local/posterior误差、相对残差、gate和方差。相反极性邻居组202条；有邻居但无非零极性冲突250条（包含零标签，不能称为严格同极性组）；无邻居5条。

## 实际过程与输入

[逐轮记录](../chsims_diagnostics_20260922/process_records.json)保留4组各50轮loss、LR、argmax指标和时序监督覆盖。约68.5%的batch没有有效时序正样本，平均每轮有效anchor计数约63。这是当前batch记录规则，不是数据全集潜在配对数量。

loss原值来自四舍五入日志；weighted CE/temporal由已知权重计算，不能当成更高精度的原始tensor。历史日志没有可提取的裁剪前梯度范数和分项梯度夹角，未补造。

实际加载英文列、RobertaTokenizerFast，max_length128；本批4组val456/test457都没有截断，输入样例及token保存在各预测目录的input_audit文件。没有更换文本或测试标签。

## 下一步边界

先完成已提交的邻居detach与分类头冻结/全模型短程对照，不叠加CDF或符号hinge。过去的gate/context、L1权重、种子和符号BCE网格不重复。上述证据区分了输出与表示问题，但没有直接测得分项梯度冲突，不能预设detach必然有效。
