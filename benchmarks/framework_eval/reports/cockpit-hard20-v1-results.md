# 困难座舱长期记忆 20 题：V1 实验报告

日期：2026-08-26  
协议：`cockpit-long-memory-20-v1`

## 1. 公开数据结论

有直接相关的公开数据，但没有一套同时覆盖车载、多会话更新、聚合、区间时间、
多乘员约束和拒答的现成困难 QA 集。

- [CarMem](https://github.com/johanneskirmayr/CarMem) 是最贴近目标的公开项目：
  面向车内语音助手长期偏好，仓库数据包含 100 位用户，每位用户 10 个偏好、
  10 段提取会话、10 条检索表达，以及每个偏好的 equal/negate/different 维护表达。
  但其主要任务是偏好提取、维护和单偏好检索，不是复杂长期记忆问答。
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 提供多会话推理、知识
  更新、时间推理、偏好和拒答能力分类，但不是智能座舱数据。

本地审计 CarMem commit
`a08f8affdecad0ed092b33e9bf68e4e1928be36e` 时，仓库没有显式 LICENSE 文件。
因此本实验没有复制或重新发布 CarMem 原句，只借鉴能力分类，生成完全原创内容。

## 2. 原创困难集

数据目录：
`benchmarks/framework_eval/challenges/cockpit_long_memory_20_v1/`

- 4 个相互隔离的驾驶员历史；
- 44 个带时间戳 session、97 条消息；
- 20 道中英文问题；
- 6 道知识更新、9 道多会话、3 道时间推理、2 道单会话偏好；
- 4 道不可回答题；
- 16 道可回答题均有 message-level evidence ID；
- 当前 `cockpit_v1` 确定性槽位编译器预检为 0/20，因此所有实验均真实调用
  回答模型。

覆盖的困难模式包括状态替换、取消/改期、有效期地点别名、条件偏好例外、
跨乘员约束交集、跨 session 频次聚合、个性化推荐和正确拒答。选择清单在运行前
冻结，题目和答案没有用于检索路由选择。

对应 Git 提交：

- `085c90b test: add hard cockpit long-memory challenge`
- `a978c15 test: freeze hard cockpit challenge selection`

## 3. 公平实验协议

- 回答模型：本地 `llama3.2:latest`，最大输出 192 token，并发 2；
- 20/20 道题均调用回答模型，确定性回答 0 次；
- Judge：LongMemEval 原任务提示词 + `deepseek-v4-flash`；
- TencentDB：4 个全新 user/agent/task 隔离 scope，L0/L1 在线构建，L2/L3 异步；
- TencentDB 和 BM25 的回答上下文上限为 2,200 字符；Full Context 作为无此上限的
  检索上界；
- 没有修改 MemoryCore 源码、MemoryCore 提取提示词、数据集答案提示词、回答
  system prompt 或 Judge 提示词/判分语义。

DeepSeek V4 默认思考曾导致 16 个 Judge 输出 token 全被 reasoning 占用、正文为空。
提交 `4159568` 只在 DeepSeek Judge 请求体增加
`"thinking":{"type":"disabled"}`，属于提供商协议兼容修复；提示词与
Yes/No 解析逻辑不变。有效评分目录均以 `-v3` 结尾；更早的 0 分目录是协议或
网络错误，不是模型分数。

## 4. 端到端结果

| 方案 | 原始 Judge | 保守核验 | 完整证据题 | 平均 evidence recall | 平均上下文字符 | 回答 prompt token | 回答总 token | 平均检索 | 平均回答 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Context | 18/20, 90% | 18/20, 90% | 16/16 | 100.0% | 3,623.0 | 30,746 | 31,210 | 0.03 ms | 453 ms |
| BM25 Top-8 | 16/20, 80% | 16/20, 80% | 11/16 | 83.85% | 1,333.2 | 13,242 | 13,674 | 0.16 ms | 267 ms |
| TencentDB 在线首答（L0/L1） | 17/20, 85% | 17/20, 85% | 9/16 | 75.0% | 968.5 | 10,432 | 10,961 | 321 ms | 407 ms |
| TencentDB L0-L3 全就绪 | 18/20, 90% | **17/20, 85%** | 9/16 | 75.0% | 1,543.8 | 19,521 | 20,081 | 294 ms | 515 ms |

主要对比：

- 在线 TencentDB 比 BM25 高 5 个百分点，同时回答 prompt token 少 21.2%；
- 在线 TencentDB 比 Full Context 少 66.1% prompt token，准确率低 5 个百分点；
- L0-L3 原始分数与 Full Context 同为 90%，prompt token 少 36.5%；
- 但 L0-L3 相比在线首答多 87.1% prompt token，稳定核验没有证明准确率提升。

四层就绪检索在 12 道 fallback 题中加入 12 个 L2 和 12 个 L3 hit，8 道 fast
题仍只读取 L0。平均上下文从 968.5 增至 1,543.8 字符；12 道题触发上下文
裁剪，最大保留 2,360 字符（2,200 预算加至多 160 字符完整行溢出）。这说明
L2/L3 不适合默认注入，应由问题类型和信息增益选择性读取。

### Judge 稳定性说明

q17 的在线与四层预测文本完全相同：回答了 9 月 10 日地址，却说 9 月 16 日地址
未知。第一次 Judge 返回 `no`，第二次对同一文本返回 `yes`；参考答案明确要求
两个日期的地址，因此第二次是 false positive。原始 90% 保留用于审计，但系统
结论采用保守的 85%。20 题是诊断集，不应把单次 Flash Judge 的 5 个百分点
波动解释为真实提升。

## 5. 记忆构建成本

选择性构建 trace：

- 44 个 session / 97 条源消息；
- 25 个 session 进入 L1，19 个只保留在线层；session 选择率 56.8%；
- 5,241 个源字符中 3,139 个进入 L1，字符选择率 59.9%；
- conversation batching 将 L1 抽取合并为 4 次模型调用；
- L2 4 次、L3 4 次，总构建模型调用 12 次；
- 每个驾驶员 L1 原子记忆数分别为 4、4、7、6，均通过构建 assurance，修复重试
  为 0。

| 构建层 | 调用 | 输入 token | 输出 token | 总 token | 平均单次模型延迟 |
|---|---:|---:|---:|---:|---:|
| L1 | 4 | 7,108 | 2,284 | 9,392 | 5.39 s |
| L2 | 4 | 10,246 | 727 | 10,973 | 17.17 s |
| L3 | 4 | 6,048 | 455 | 6,503 | 9.23 s |
| 合计 | 12 | 23,402 | 3,466 | **26,868** | — |

以上是 MemoryCore 日志报告的 LLM token，不含 embedding 服务 token/计费。
L2 在 L1 后延迟 180 秒异步触发，不阻塞在线首答；四个 L3 persona 最终均生成，
队列为 0，任务失败为 0。

## 6. Bad case 追溯

### q06：跨 session 频次聚合

问题要求从三个充电事件统计最常去的站点及次数。自适应路由以
`moderate_coverage_with_margin` 在 Top-1 提前结束，只返回“虹桥枢纽”一次；
正确答案需要合并另外两个“徐汇滨江超充站”事件。BM25 和 Full Context 都拿到
三条 evidence。根因是 aggregate/frequency 问题误走 fast path，不是 L1 抽取
失败。

### q14：预约更新与改期

问题包含 “finally scheduled”，但路由以 `high_lexical_coverage` 返回最初的
“July 28 at 9 a.m.”，漏掉后续改到 “July 30 at 2 p.m.”。L1 已正确构建一条
包含原预约和改期的 compact memory，但 fast path 没有查询 L1。根因是英文更新
触发词覆盖不足。

### q17：双日期有效期别名

问题同时问 9 月 10 日和 9 月 16 日的 `work` 地址。Top-1 只返回 9 月 1–14 日
的 Riverside Office，漏掉 9 月 15 日恢复 Harbor Lab 的更新。L1 已抽取恢复
事件，L2/L3 也已生成，但 fast path 不读取它们。根因是多时间点/区间比较没有
强制 fallback；此外该题暴露了 Flash Judge 的 Yes/No 波动。

这三个错误共同说明：当前困难集上的首要优化点是通用 query-risk gate，而不是
继续扩大默认上下文或重写 MemoryCore 构建提示词。应让 aggregation、final/current
update、respectively/two-time-span 等高风险查询进入 Top-3 + L1；再用新的 held-out
困难集验证，不能在这 20 道开发题上调完后直接声称泛化提升。

## 7. 可审计产物

- 题库说明：`benchmarks/framework_eval/challenges/cockpit_long_memory_20_v1/README.md`
- 冻结问题：`benchmarks/framework_eval/challenges/cockpit_long_memory_20_v1/questions.jsonl`
- 冻结选择：`benchmarks/framework_eval/challenges/cockpit_long_memory_20_v1/selection.json`
- 运行根目录：`benchmarks/framework_eval_runs/cockpit-hard20-20260826-v1/`
- TencentDB 在线预测：`tencentdb-k8-c2200/predictions-llama32.jsonl`
- TencentDB 四层预测：`tencentdb-k8-c2200/predictions-l0l3-ready-llama32.jsonl`
- 有效 Judge：各方案 `score-*-v3/judge-results.jsonl`

本结果只能作为 20 题工程诊断，不能替代 300–500 条 held-out 或官方 benchmark
分数。
