# ECG-5001 逐搏探索：方法纠错与两阶段重跑



本轮纠错不以提高AUROC为目标。旧输出保留；原始数据只读，fold9/10波形未打开，未做GARCH/EGARCH。



## 撤回与来源

旧报告的“训练折模板”“B为纯无序统计”“shuffle支持顺序无收益”三项论证撤回。旧A/B/C含标签模板泄漏，旧B含相邻差，shuffle未重算全部C且将5×8折SD误当置换不确定性，故旧性能只作有缺陷的历史数字。

PDF来自 regenerate_ascii_casebook.py 的英文PNG，再由 make_casebook_pdf.py 合成，红线使用pk[i]/500；run_exploration.py的(src+1)/500不是该PDF最终红线来源。报告/v_h表无独立生成入口，当前文件哈希不能冒充历史运行哈希。详见 SOURCE_AUDIT.md、source_provenance.csv。



## 两阶段与分析单位

阶段一精确重放旧检测与切拍，并逐条比对final5代表性心拍。阶段二在原候选峰上加入预先记录的低幅低斜率候选筛查：前一保留峰后200–400ms，幅度<45%、±40ms局部差分峰值<50%时删除。250ms候选峰距和RR门槛均未抬高；强弱条件不满足时保留短RR。仍可能删掉真实低幅/宽QRS，也可能漏掉错误峰。该规则利用当前样本反馈，属于开发探索，未独立确认准确率。

每位患者一条记录。主对照300患者，失败记录特征在训练折内填补；共同有效患者敏感性重新在两阶段共同患者上训练/验证。严格方案只允许训练折prototype_pool且12导联v_h<0.3，每类至少20，缺少时不补验证数据，不报全体CV。



| stage | label | n | success | beats |
|---|---|---|---|---|
| stage1 | LVH | 150 | 146 | 1434 |
| stage1 | NORM | 150 | 139 | 1385 |
| stage2 | LVH | 150 | 150 | 1432 |
| stage2 | NORM | 150 | 149 | 1400 |



## 合并折外性能（原始顺序）

| stage | cohort | scheme | model | n_evaluated | n_missing | auroc | balanced_accuracy | coverage |
|---|---|---|---|---|---|---|---|---|
| stage1 | all300 | exploratory | A | 300 | 0 | 0.8441 | 0.7867 | complete |
| stage1 | all300 | exploratory | B | 300 | 0 | 0.8503 | 0.8067 | complete |
| stage1 | all300 | exploratory | C | 300 | 0 | 0.8484 | 0.8100 | complete |
| stage1 | all300 | strict | A | 67 | 233 | 0.8705 | 0.8610 | partial_not_full_CV |
| stage1 | all300 | strict | B | 67 | 233 | 0.8810 | 0.8729 | partial_not_full_CV |
| stage1 | all300 | strict | C | 67 | 233 | 0.8819 | 0.8729 | partial_not_full_CV |
| stage1 | common_valid | exploratory | A | 285 | 0 | 0.8499 | 0.7936 | complete |
| stage1 | common_valid | exploratory | B | 285 | 0 | 0.8585 | 0.8008 | complete |
| stage1 | common_valid | exploratory | C | 285 | 0 | 0.8563 | 0.8044 | complete |
| stage1 | common_valid | strict | A | 63 | 222 | 0.8826 | 0.8565 | partial_not_full_CV |
| stage1 | common_valid | strict | B | 63 | 222 | 0.8967 | 0.8690 | partial_not_full_CV |
| stage1 | common_valid | strict | C | 63 | 222 | 0.8946 | 0.8908 | partial_not_full_CV |
| stage2 | all300 | exploratory | A | 300 | 0 | 0.8592 | 0.8167 | complete |
| stage2 | all300 | exploratory | B | 300 | 0 | 0.8688 | 0.8200 | complete |
| stage2 | all300 | exploratory | C | 300 | 0 | 0.8695 | 0.8167 | complete |
| stage2 | all300 | strict | A | 226 | 74 | 0.8871 | 0.8482 | partial_not_full_CV |
| stage2 | all300 | strict | B | 226 | 74 | 0.8936 | 0.8524 | partial_not_full_CV |
| stage2 | all300 | strict | C | 226 | 74 | 0.8945 | 0.8570 | partial_not_full_CV |
| stage2 | common_valid | exploratory | A | 285 | 0 | 0.8556 | 0.8043 | complete |
| stage2 | common_valid | exploratory | B | 285 | 0 | 0.8655 | 0.8008 | complete |
| stage2 | common_valid | exploratory | C | 285 | 0 | 0.8665 | 0.8082 | complete |
| stage2 | common_valid | strict | A | 100 | 185 | 0.8826 | 0.8662 | partial_not_full_CV |
| stage2 | common_valid | strict | B | 100 | 185 | 0.8777 | 0.8456 | partial_not_full_CV |
| stage2 | common_valid | strict | C | 100 | 185 | 0.8777 | 0.8456 | partial_not_full_CV |



严格方案若只覆盖部分折，上表partial值只是可评估子集，不能与全300探索方案直接排名。



## 配对差与顺序置换

| stage | cohort | comparison | n | delta_auroc | auroc_low | auroc_high | delta_balanced_accuracy |
|---|---|---|---|---|---|---|---|
| stage1 | all300 | B-A | 300 | 0.0063 | -0.0068 | 0.0197 | 0.0200 |
| stage1 | all300 | C-B | 300 | -0.0019 | -0.0111 | 0.0054 | 0.0033 |
| stage1 | common_valid | B-A | 285 | 0.0086 | -0.0054 | 0.0222 | 0.0072 |
| stage1 | common_valid | C-B | 285 | -0.0022 | -0.0115 | 0.0059 | 0.0036 |
| stage2 | all300 | B-A | 300 | 0.0095 | -0.0050 | 0.0223 | 0.0033 |
| stage2 | all300 | C-B | 300 | 0.0008 | -0.0055 | 0.0068 | -0.0033 |
| stage2 | common_valid | B-A | 285 | 0.0099 | -0.0052 | 0.0227 | -0.0034 |
| stage2 | common_valid | C-B | 285 | 0.0010 | -0.0059 | 0.0072 | 0.0074 |



| stage | cohort | n_evaluated | auroc_mean | auroc_sd_over_5_repetitions | auroc_min | auroc_max |
|---|---|---|---|---|---|---|
| stage1 | all300 | 300 | 0.8482 | 0.0028 | 0.8450 | 0.8515 |
| stage1 | common_valid | 285 | 0.8564 | 0.0029 | 0.8527 | 0.8593 |
| stage2 | all300 | 300 | 0.8663 | 0.0021 | 0.8642 | 0.8689 |
| stage2 | common_valid | 285 | 0.8618 | 0.0020 | 0.8594 | 0.8648 |

每次置换先合并8折患者预测计算AUROC/BA，再对5次指标汇总；与原始C比较也使用合并OOF。A/B特征及预测逐折逐次保持不变。paired_bootstrap.csv给B−A、C−B及original_C−shuffle_C的配对区间，1000次患者级分层重采样。它固定折外预测，不包含重新训练造成的全部不确定性；5次置换也不足构造精细置换检验。



## 波形与质量

旧B_stepamp的4个跨缺失连接（4481、11404、14351、17652）已留表并排除。casebook覆盖全部15条旧失败、全部旧成功例和新增确定性NORM/LVH成功对照；局部图包含真实原始信号、带通composite、检测实际阈值信号（平方导数120ms平滑能量）、能量候选与精化峰。原casebook只显示composite，未显示实际阈值能量，不能把两者混称。

图像初审与算法输出不是人工金标准；manual_review_form.csv的人审核字段空白、状态未审核。失败时候选边界和最终接受心拍边界分开。归一化心拍用采样索引，不映射成固定生理秒。

SQI在0.0005–0.005mV、比例阈值0.20下记录标志率接近饱和，撤回“标志率明显敏感”。逐导联表进一步列连续低差分时长、最长严格重复值时长、众数比例、非QRS低斜率比例，不能据此认定全是假阳性或全不合格。SQI仍未用于排除。



## 支持与不支持的结论

新结果只支持本开发样本和当前特征/模型下的增量大小，不能证明时序必然有用或不存在。需要结合C−B配对区间、原始C与五次置换分布，而不是单一最高AUROC。两阶段共同有效患者结果有助区分检测修改与样本组成，但检测器已受本样本视觉反馈，仍不是独立测试。

下一步优先独立标注全部疑似额外峰及成功对照，冻结检测规则后在未读fold9/10另行预注册验证；年龄、性别、噪声、心率仍可能解释组间差异，本轮不作因果或临床归因。



## 复现

从 code/fixed_pipeline.py 用 --output 指向全新空目录；README.md给出命令。所有模板、逐拍缓存、特征缺失表、完整OOF及汇总验证均保存。代码哈希、依赖版本和原始输入/旧输出前后哈希随结果交付。