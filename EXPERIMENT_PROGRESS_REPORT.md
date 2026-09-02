# TencentDB Agent Memory 实验进展与结果分析

更新时间：2026-08-06 06:26 UTC（断点续跑启动时）

## 1. 当前结论

TencentDB Agent Memory 已经完成独立 Gateway、长期记忆 L0-L3、BM25/Embedding 混合召回，以及短期记忆 Offload Stage A-G 的工程验证。LoCoMo 上已经有完整的 1382 题三臂对照结果；LongMemEval 的 500 题实验已按 16 个分片落盘，上一轮因 Token Plan 额度耗尽只完成了部分问题，现已使用 `qwen3.8-max` 基于原结果断点续跑。

LongMemEval 的当前运行不是重新建库：每个分片的 `locomo_runner` 检测已有结果，只补充未成功的 Trial；成功结果、Store、Token Ledger 和日志均保留。

## 2. 系统链路

```text
原始多 Session 对话
  -> L0 原始 JSONL
  -> L1 LLM 结构化事实/事件记录
  -> L2 LLM 场景块聚合
  -> L3 LLM 用户画像增量综合
  -> BM25 + Embedding + RRF 召回
  -> 主 Agent 回答
  -> LLM Judge / F1 / BLEU
```

短期记忆链路为：工具输出 -> Inline/Offload 判断 -> LLM 压缩与原文 SHA-256 保存 -> Working Context/Steps/Mermaid -> 主 Agent 按 `node_id` 下钻 -> Checkpoint 与跨进程恢复。

## 3. LoCoMo 定量结果

### 3.1 4 题工程 Gate

| 指标 | 结果 |
|---|---:|
| 问题数 | 4 |
| Arm 数 | 3 |
| 预期 Trial | 12 |
| 实际 Trial | 12 |
| paired-valid | 4/4 |
| 错误/超时 | 0/0 |

该 Gate 证明流程完整，不代表大规模准确率。

### 3.2 48 题完整三臂对照

| Run | Full Context LLM | BM25 LLM | TDAI LLM | Full F1 | BM25 F1 | TDAI F1 | TDAI 输入 Token 节省 |
|---|---:|---:|---:|---:|---:|---:|---:|
| q48-eval-v1 | 75.00% | 33.33% | 58.33% | 0.5536 | 0.2425 | 0.4420 | 81.00% |
| q48-general-v1 | 77.08% | 31.25% | 54.17% | 0.5833 | 0.2010 | 0.3915 | 80.83% |

`q48-general-v1` 的 BLEU 分别为 Full Context 0.5213、BM25 0.1586、TDAI 0.3417。48 题实验中，TDAI 稳定地显著优于 BM25，但仍低于 Full Context。

### 3.3 48 题检索策略实验

| Run | TDAI LLM | TDAI F1 | Token 节省 |
|---|---:|---:|---:|
| q48-multihop-v2 | 58.33% | 0.4445 | 81.06% |
| q48-neighbors-v3 | 58.33% | 0.4784 | 78.67% |
| q48-adaptive-v4 | 62.50% | 0.4825 | 78.81% |
| q48-adaptive-v5 | 60.42% | 0.4645 | 78.83% |
| q48-adaptive-v6 | 62.50% | 0.4782 | 78.93% |

这些是同一 48 题集合上的实验配置比较，不能直接视为已经泛化到所有数据集的最终配置。

### 3.4 1382 题完整三臂结果

| Arm | 题数 | LLM Judge | F1 | BLEU | 输入 Prompt Token | 平均/题 |
|---|---:|---:|---:|---:|---:|---:|
| Full Context | 1382 | 75.54% | 0.6113 | 0.5434 | 32,082,336 | 23,214 |
| BM25 | 1382 | 34.59% | 0.3027 | 0.2587 | 608,155 | 440 |
| TDAI Memory | 1382 | 51.52% | 0.4513 | 0.3965 | 5,981,279 | 4,328 |

TDAI 相对于 Full Context 的输入 Prompt Token 节省为 **81.36%**；相对于 BM25，TDAI 的准确率高 **16.93 个百分点**，但相对于 Full Context 仍低 **24.02 个百分点**。

### 3.5 LoCoMo 结果解释

Full Context 是准确率上限参考，但平均每题约 2.3 万输入 Token。BM25 只使用约 440 Token/题，成本最低但语义覆盖不足。TDAI 约 4,328 Token/题，在节省 81% Token 的同时，准确率明显高于 BM25，说明 L1/L2/L3 和混合召回确实提供了超越纯关键词检索的能力。

目前 TDAI 未超过 Full Context，主要损失来自证据遗漏、多跳跨 Session 组合、时间关系恢复和主 Agent 对召回证据的组织，而不是单纯答案格式问题。

## 4. LongMemEval 断点续跑

运行目录：

`benchmarks/production_runs/results-longmemeval/longmemeval-full500-v1`

分片数：16；问题总数：500；三个 Arm：Full Context、BM25、TDAI Memory。

### 4.1 断点前快照

| Arm | 成功回答 | 成功率（以 500 题计） |
|---|---:|---:|
| Full Context | 347 | 69.40% |
| BM25 | 348 | 69.60% |
| TDAI Memory | 347 | 69.40% |

三臂同时成功的问题为 345 个，至少一个 Arm 成功的问题为 349 个。上述是“回答生成成功率”，不是 Judge 正确率。

上一轮停止的直接原因是 Token Plan 5 小时额度耗尽，另有少量连接错误和上游 `data_inspection_failed`。因此这批错误不能直接判为记忆或回答错误。

### 4.2 当前续跑

本次续跑已于 06:26 UTC 启动，16 个分片均已进入 Runner；当前处于前几个问题的 Gateway 恢复/校验阶段。后续成功数、paired-valid 和额度错误会继续写入原分片目录。

断点续跑命令使用：

```bash
TDAI_ANSWER_LLM_MODEL=qwen3.8-max \
TDAI_MEMORY_LLM_MODEL=qwen3.8-max \
python benchmarks/production/longmemeval_parallel.py \
  --run-prefix longmemeval-full500-v1 --shards 16 --base-port 8500
```

`longmemeval_parallel.py` 在已有分片目录存在时自动给每个 Runner 加 `--resume`。因此不会删除原结果，也不会重建已经完成的 Store。

### 4.3 LongMemEval 已有 Judge 结果

目前全量 500 题 Judge 尚未完成。已完成的 93 题 Judge 快照为：

| Arm | 正确/总数 | Judge 准确率 |
|---|---:|---:|
| Full Context | 68/93 | 73.12% |
| BM25 | 51/93 | 54.84% |
| TDAI Memory | 58/93 | 62.37% |

该 93 题诊断显示：TDAI 平均 Evidence Session Recall 为 0.8407，BM25 为 0.8747，Full Context 为 1.0。TDAI 的主要问题是证据没有进入召回上下文，而不是所有问题都由 LLM 答案生成失败造成。

## 5. 分阶段问题定位

现有诊断将 93 个可 Judge 问题划分为：

| 诊断 | 数量 |
|---|---:|
| TDAI 正确 | 58 |
| 答案或上下文选择差距 | 12 |
| 困难问题/答案生成差距 | 5 |
| TDAI 检索回归 | 5 |
| 证据覆盖缺口 | 13 |

LongMemEval 分类上，TDAI 在 single-session-preference 为 100%，single-session-assistant 为 85.71%，single-session-user 为 81.82%；multi-session 为 54.17%，temporal-reasoning 为 45.16%。这说明当前通用优化重点应放在多 Session 证据覆盖和时间线重建，而不是针对单一题型增加硬编码。

## 6. Judge 和指标口径

- LoCoMo 使用官方 Refined evaluator，当前 Judge 模型为 Qwen3-14B，同时记录 `llm_score`、F1、BLEU；LLM 分数大于 0.5 才计为通过。
- LongMemEval 使用任务类别专用 Yes/No Judge Prompt，当前 Judge 模型也是 Qwen3-14B；只有全量回答成功并完成 Judge 后才能报告正式准确率。
- Token 节省率只比较三臂回答阶段的输入 Prompt Token，不把离线建库 Token 混入；因此它衡量的是推理阶段上下文压缩收益。
- `Evidence Session Recall` 衡量标准答案来源 Session 是否被召回，不等同于事实级覆盖率；它用于定位召回问题，不能直接替代最终 Judge。

## 7. 下一步验收标准

1. 断点续跑补齐 500 个问题的三臂成功回答。
2. 统计 `paired-valid=500/500`、错误为 0 或明确列出不可恢复错误。
3. 对完整三臂结果运行 LongMemEval Judge。
4. 输出总体准确率、各类别准确率、Evidence Recall、Token 和失败原因。
5. 对 TDAI 错题按“L1 缺失、召回缺失、多跳组合失败、时间线错误、答案生成错误”分类。

只有完成以上步骤，才能判断 TencentDB Memory 是否在 LongMemEval 上具备跨数据集泛化优势。

## 8. 全部对比实验与消融实验记录

本节集中记录已经实际运行过的对比，而不是只保留当前最好的一组数字。

### 8.1 工程 Gate 与正式质量评测的区别

| 实验 | 目的 | 数据规模 | 结论 |
|---|---|---:|---|
| `gate4q-v1` 至 `gate4q-v8` | 验证 Gateway、建库、召回、回答、配对和恢复 | 4 题 × 3 Arm | Gate 通过；不作为质量排名 |
| `q48-v1` | 生成固定 48 题完整 Store 和基准产物 | 48 题 | 主要用于准备和后续复用 Store |
| `q48-eval-v1` | 三臂正式基线 | 48 题 | Full 75.00%、BM25 33.33%、TDAI 58.33% |
| `q48-general-v1` | 通用检索改动后的独立复测 | 48 题 | Full 77.08%、BM25 31.25%、TDAI 54.17% |
| `full1382-v1` | LoCoMo 全量三臂质量评测 | 1382 题 | Full 75.54%、BM25 34.59%、TDAI 51.52% |

4 题 Gate 的完整记录为 12/12 Trial、4/4 paired-valid、0 错误、0 超时。它证明工程链路完整，但由于题数很少且 Gate 只用于快速验收，不能与 1382 题准确率混用。

### 8.2 LoCoMo 基线与三臂对照

三臂始终使用同一题目、同一标准答案和同一回答模型，区别只在提供给回答模型的上下文：

```text
Full Context：完整对话历史
BM25：关键词检索结果
TDAI Memory：L0/L1/L2/L3 + BM25/Embedding 混合记忆
```

1382 题结果表明：

- Full Context 的准确率最高，但每题约 23,214 输入 Token。
- BM25 每题约 440 输入 Token，但准确率只有 34.59%。
- TDAI 每题约 4,328 输入 Token，节省 81.36%，准确率为 51.52%。
- TDAI 比 BM25 高 16.93 个百分点，说明分层记忆和语义召回确实有效。
- TDAI 比 Full Context 低 24.02 个百分点，说明证据覆盖、多跳组合和时间线恢复仍是主要瓶颈。

### 8.3 固定 48 题的检索消融

| 版本 | 改动 | TDAI Judge | TDAI F1 | 结果解释 |
|---|---|---:|---:|---|
| `q48-eval-v1` | 单查询、孤立 L0 消息的稳定基线 | 58.33% | 0.4420 | 作为消融起点 |
| `q48-multihop-v2` | 有限查询分解、原问题保留、coverage-first RRF | 58.33% | 0.4445 | Judge 不变，F1 极小提升 |
| `q48-neighbors-v3` | 所有问题都扩展 L0 前后邻域和时间锚点 | 58.33% | 0.4784 | 证据重叠增加，但噪声抵消收益 |
| `q48-adaptive-v4` | 只对时间/原因/属性等问题启用邻域扩展 | 62.50% | 0.4825 | 当前 48 题最佳之一 |
| `q48-adaptive-v5` | 在 v4 上加入更强列表/集合回答提示 | 60.42% | 0.4645 | 出现回归，提示被撤回 |
| `q48-adaptive-v6` | v4 检索策略 + 原稳定回答提示 | 62.50% | 0.4782 | 最终保留候选版本 |

逐题对比中，v4 相对基线为 3 胜、1 负、44 平；因此 4.17 个百分点的提升来自少数需要邻域证据的问题，并不是所有问题都普遍改善。

### 8.4 `q48-general-v1` 为什么单独记录

`q48-general-v1` 是在通用证据覆盖源码改动后重新运行的独立实验，不是 v2-v6 消融链中的严格单变量节点。它包含动态 L1/L0 上下文和部分多查询行为，实际 Trial 中有 35 个普通 Hybrid、12 个三路 Multi-query、1 个四路 Multi-query。

它的结果低于 `q48-eval-v1`，说明“增加通用检索能力”不等于在小样本上必然提升；更多证据可能带来排序变化或噪声。这个结果被保留，作为反例和泛化风险记录，而没有被隐藏。

### 8.5 已尝试但未采用的优化方向

#### A. Embedding 混合召回

最初发现 Gateway 配置中 Embedding 服务没有真正启用，导致系统退化为 Keyword/BM25。随后补充了：

- Embedding API 探针
- 向量写入和维度检查
- 429 重试、退避和降突发
- BM25 + Embedding + RRF
- 多查询 Hybrid 召回

这是基础设施修复，不是针对 LoCoMo 的答案硬编码。

#### B. 单纯增加 L1/L2/L3 上下文

曾尝试提高长期记忆层的召回范围和上下文预算。结果是部分问题证据覆盖增加，但同时上下文变长、无关事实增加，最终答案 Judge 没有稳定改善。因此当前策略倾向于“证据覆盖优先，但有界预算”，而不是把所有 L1/L2/L3 全部塞给回答模型。

#### C. 多跳查询分解

采用模型无关的有限分解：保留原始问题，最多生成少量子查询，再做 Coverage-first RRF。它解决了一个问题包含多个事实时的单查询漏召回，但在 48 题上的 Judge 只从 58.33% 变为 58.33%，说明还需要更好的多跳证据去重和关系拼接。

#### D. 全量邻域扩展

对每个命中 L0 消息都加入前后消息。F1 上升，但 Judge 不变，且 Token 节省从约 81% 降至约 78.7%。因此没有采用“所有问题都扩展”的默认策略。

#### E. 自适应邻域门控

根据问题是否涉及时间、原因、属性和事件关系决定是否扩展邻域；计数、Yes/No 和固定集合问题不默认扩展。该方法在 48 题 v4/v6 达到 62.5%，是当前候选方向，但仍需 LongMemEval 和其他数据集验证。

#### F. 更强的集合/列表回答提示

尝试要求模型对列表、计数和集合问题进行更强的完整性检查。`q48-adaptive-v5` 出现回归，故已撤回。当前继续使用经过回归验证的最小证据回答提示。

#### G. 强制 Unknown / 过度约束答案

曾尝试在证据不足时更激进地输出 Unknown，并对某些集合问题使用专用答案格式。由于会把“证据不完整但可以部分作答”的情况错误判成不可回答，出现回归，未作为默认生产策略。

#### H. 提取密度直接增大

没有简单地无限增加 L1 记录数量。L1 提取过密会把推断、重复和低置信度内容带入召回，造成噪声；过稀则会丢失事实。当前方向是保留来源消息、时间、Session 和优先级，结合 L2 场景聚合与证据覆盖诊断来优化，而不是针对 LoCoMo 增加固定条数。

### 8.6 LongMemEval 对比和诊断实验

已运行或落盘的 LongMemEval 实验包括：

| 实验 | 目的 | 结果状态 |
|---|---|---|
| `longmemeval-gate1-v1/v2` | 单样本 Gateway、建库和回答 Gate | Gate 通过 |
| `longmemeval-gate1-batch5-v1` | 多样本小批量稳定性 | Gate 通过 |
| `longmemeval-gate1-capturebatch5-v2` | 捕获批处理和 Store 完整性 | Gate 通过 |
| `longmemeval-optimized-reg32-v1` | 优化策略困难集回归 | 32 题，未接受为最终配置 |
| `longmemeval-full500-v1` | 500 题三臂正式实验 | 曾因额度中断，正在断点续跑 |
| `longmemeval-full500-v1-diagnostics` | 证据覆盖、答案生成和检索回归归因 | 已完成 93 题诊断 |

32 题困难回归集结果：Full Context 20/32（62.5%）、BM25 12/32（37.5%）、TDAI 8/32（25.0%）。该结果说明某些优化组合会在困难题上回归，因此没有直接把它们设为默认配置。

已有 93 题 Judge 结果：Full Context 73.12%、BM25 54.84%、TDAI 62.37%。诊断将问题分为：TDAI 正确 58、答案或上下文选择差距 12、困难/答案生成差距 5、检索回归 5、证据覆盖缺口 13。

### 8.7 短期记忆消融与工程验证

短期记忆不是只做了一个 Offload API，而是按 Stage A-G 验证：

| Stage | 验证内容 | 定量结果 |
|---|---|---|
| A | 文件存储、幂等、SHA-256、损坏 JSONL 恢复 | 通过 |
| B | 可逆压缩 Mock 验证 | 字符减少 94.43%，原文恢复通过 |
| C | 独立 Mock Gateway | 鉴权、隔离、重启恢复通过 |
| D | 真实 MiniMax 压缩 | 字符减少约 92%-95%，原文可恢复 |
| D.2 | Evidence Guard 语义去重 | 8/8 回归测试通过 |
| E | 自动 Inline/Offload | 4,994 字符压缩至 378 字符，减少 92.43%；事件幂等通过 |
| F | DeepSeek 自主下钻 | node_id、SHA-256、精确证据提取通过 |
| G | 生产 Runner | StageG_PASS=True，跨进程 Resume 通过 |

Stage G 曾记录 DeepSeek 使用 3,744/7,000 Token、MiniMax 使用 2,794/4,000 Token，且无 Mock Fallback。

## 9. 当前优化优先级

基于全部对比实验，当前不再优先增加更多 Prompt 约束，而是按以下顺序推进：

1. 补齐 LongMemEval 500 题三臂结果和全量 Judge。
2. 对 TDAI 错题区分 L1 抽取缺失、向量/BM25 未召回、邻域不足、多跳关系断裂和答案生成错误。
3. 优先优化多 Session 和 temporal-reasoning，而不是针对 LoCoMo 某个类别调参。
4. 保留自适应邻域和有限多跳作为候选，通过 LongMemEval 和其他数据集做 paired A/B。
5. 继续使用有界上下文，避免用增加 Token 换取不可控噪声。
6. 只有在跨数据集稳定提升后，才把候选策略设为生产默认配置。

## 10. 局部根因排查记录（conv-26/conv-43）

为避免继续消耗大规模额度，准备了一个不使用 q48 选题的局部测试集：`conv-26` 和 `conv-43` 各 8 题，共 16 题、三臂 48 次回答，复用已有 `q48-v1` Store，不重新调用 L1/L2/L3 提取模型。

首次启动尚未进入模型调用，Runner 在 Gateway 启动阶段失败：`npx tsx` 尝试创建 `/tmp/tsx-1000/*.pipe` IPC 管道时收到 `EPERM`。这不是记忆召回或答案质量问题，也没有消耗本轮答案 Token。直接使用 `node --import tsx` 可以绕过 `tsx` IPC 创建步骤；后续局部实验应优先修复或覆盖这个测试启动方式，再继续模块消融。

已有落盘结果对这两个 Conversation 的无 Token 诊断显示：

- 当前 TDAI 基线每题通常返回 8 条 L1 和 6 条 L0，说明不是“完全没有召回结果”。
- `q48-general-v1` 和后续版本在部分问题上触发了 `multi_query`，但上下文中仍有大量“不足证据/Unknown”答案。
- `q48-adaptive-v6` 将 L0 从通常 6 条扩展到约 9-15 条，部分答案变得更精确，但也增加了上下文长度；这解释了 F1 上升而 Judge 未全面上升。
- `full1382-v1` 的 TDAI 平均上下文约 4,328 输入 Token，而 Full Context 约 23,214 Token；当前差距首先表现为证据覆盖和多跳组合差距，而不是单纯 Token 预算差距。

下一轮局部测试将按以下顺序执行：

```text
Full Context
  vs BM25
  vs 当前 Hybrid（L1/L2/L3 + L0）
  vs 关闭 multi-hop 的 Hybrid
  vs 只保留 L1/L2/L3
  vs 只保留 L0
```

每一层只使用 16 个局部问题，记录答案、召回文本、L1/L2/L3 命中、L0 命中、上下文 Token 和 Judge 结果。这样可以判断损失发生在提取、索引、召回、上下文拼接还是最终回答阶段。

## 11. 当前 L1 提取密度快照

密度按“L1 JSONL 记录数”统计，不把 L2 场景块或 L3 Persona 字数当作 L1 条数。对已经完整建库的 LoCoMo `q48-v1` Store，排除一个额度中断的无效 Store 后：

| 指标 | 汇总 |
|---|---:|
| 有效 Conversation Store | 10 |
| L0 消息 | 5,882 |
| L1 记录 | 869 |
| Session | 272 |
| L1 / 消息 | 14.77% |
| L1 / 1,000 条消息 | 147.74 |
| L1 / Session | 3.19 |
| 平均每条 L1 覆盖的 L0 消息 | 约 6.77 |

各 Conversation 的 L1/Session 范围为 2.32-3.74，L1/1,000 消息范围为 114.71-169.45。比如 `conv-43` 有 680 条 L0、78 条 L1，即 11.47%、2.69 条/Session。

当前 Gateway 配置中单 Session 的 `maxMemoriesPerSession=20` 是上限，不是实际目标密度；实际模型通常只生成约 2-4 条 L1/Session。这说明提取策略偏“高置信度、宁缺毋滥”，不是每条消息都生成一条记忆。

LongMemEval 上一轮被额度中断的分片不适合用于估计真实密度：大量 Store 的 `records` 为空或未完成，观测到的低密度主要反映 API 失败和流水线未完成，而不是正常 L1 提取能力。
