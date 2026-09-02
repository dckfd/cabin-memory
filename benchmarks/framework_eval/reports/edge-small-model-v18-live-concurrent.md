# TencentDB 座舱端侧记忆优化 V18：全新 Live 建库、直达检索与并发写入

日期：2026-08-26

分支：`experiment/cockpit-memory-v13-20260824`

## 1. 结论

V18 补齐了 V17 尚未完成的线上链路验证：使用全新 namespace 完成 KVRET 相对时间
座舱任务 300 条的 L0/L1/L2/L3 建库、检索、回答和确定性评分。结果为：

- Evidence recall：300/300；
- Exact：297/300（99.00%）；
- Contains：298/300（99.33%）；
- 95 条走 source-grounded typed episode 直达，205 条走时间索引快路；
- 动态 ANN 调用和 L1/L2/L3 在线回退均为 0；
- 回答模型调用为 0，回答 prompt/completion/total token 为 0/0/0；
- 最终平均检索延迟 17.75 ms，P95 31.66 ms；
- 平均检索上下文 524.19 字符，截断 12/300；
- L0 并发写入把 8 个 conversation 的累计 ingest 从 291.90 s 降至 79.64 s，
  降低 72.72%；完整前台构建加首次检索命令从 425.80 s 降至 220.36 s，降低
  48.25%。

本轮没有修改 TencentDB MemoryCore 源码或其抽取提示词，没有修改 KVRET 数据、回答
系统提示词或 Judge。代码变更仍只位于评测适配层、确定性构建/检索编排、运行配置和测试。

## 2. Live 运行契约

两轮均使用冻结的同一批 KVRET 300 个 QA；它们实际关联 8 个 conversation、336 个
source session 和 1,860 条 source message。每轮使用独立的新 namespace，避免复用旧记忆。

选择性构建结果在两轮完全一致：

| 构建项 | 数值 |
| --- | ---: |
| source session | 336 |
| source message | 1,860 |
| L0 写入 message | 1,739 |
| 进入 L1 的 session | 31（9.226%） |
| 抑制 L1 的 session | 305 |
| source 字符 | 81,344 |
| 进入 L1 的字符 | 6,892（8.473%） |
| 类型化 navigation episode | 112 |

31 个选中 session 按 conversation 压缩为 8 个正常 L1 batch。L2/L3 采用 dirty-event
异步调度，不阻塞 L0/L1 在线写入。最终全部 scope 均可读取 L1，L2/L3 队列清空，未观察到
worker failure、retry 或 dead letter。

运行中的 MemoryCore 服务实际配置为 DeepSeek `deepseek-chat`。适配层配置中的
`MEMORY_LLM_MODEL` 不会动态改写已经启动的服务，因此本报告不把该轮误称为特定
Flash 版本。

## 3. 端到端结果

### 3.1 全新基线与最终链路

| 指标 | Live 串行基线 | V18 最终链路 | 变化 |
| --- | ---: | ---: | ---: |
| Evidence recall | 300/300 | 300/300 | 不变 |
| Exact | 297/300 | 297/300 | 不变 |
| Contains | 298/300 | 298/300 | 不变 |
| Typed 直达 / 时间快路 / 回退 | 0 / 294 / 6 | 95 / 205 / 0 | 回退清零 |
| 动态 ANN 调用 | 12 | 0 | -100% |
| 平均检索 | 32.21 ms | 17.75 ms | -44.90% |
| P50 检索 | 20.51 ms | 22.41 ms | +9.24% |
| P95 检索 | 36.94 ms | 31.66 ms | -14.31% |
| 最大检索 | 600.66 ms | 120.78 ms | -79.89% |
| 平均上下文 | 720.25 字符 | 524.19 字符 | -27.22% |
| 上下文截断 | 37 | 12 | -67.57% |
| 回答模型调用 / token | 0 / 0 | 0 / 0 | 不变 |

P50 略升是混合分布导致：95 条亚毫秒级 typed 直达拉低平均值，但第 150 条仍落在时间
索引路径。P95、最大值和平均值均下降，不能只看单个 P50 判断退化。

类别评分为 Calendar 99/100、Navigation 98/100、Weather 100/100（Exact）；Contains
分别为 99/100、99/100、100/100。

### 3.2 同库配对：typed 直达的独立收益

为排除新 namespace、服务负载和建库随机性的影响，又在同一完整 store 上配对关闭/开启
typed 直达：

| 指标 | Control | Typed direct | 变化 |
| --- | ---: | ---: | ---: |
| 平均检索 | 40.34 ms | 24.21 ms | -39.98% |
| P50 | 29.27 ms | 19.41 ms | -33.67% |
| P95 | 48.62 ms | 34.38 ms | -29.29% |
| Navigation 平均 | 31.26 ms | 1.52 ms | -95.13% |
| Navigation P95 | 44.59 ms | 1.02 ms | -97.72% |
| 平均上下文 | 729.52 字符 | 557.11 字符 | -23.63% |
| 截断 | 42 | 18 | -57.14% |

两侧 Evidence recall 与最终评分均相同。该配对更适合归因 typed 直达本身；上一表的全新
端到端对比还同时包含引号时间分类修复和正常的运行噪声。

## 4. 构建延迟与 token

### 4.1 L0 并发写入

最终实现保留每个 source session 独立的 transport ID 和消息边界，只把互不依赖的
L0-only 请求以并发 4 flush；选中进入 L1 的 batch 仍串行，避免共享 agent 锁冲突。

| 指标 | 串行基线 | 并发 4 | 变化 |
| --- | ---: | ---: | ---: |
| 8 conversation ingest 累计 | 291.90 s | 79.64 s | -72.72% |
| 单 conversation 平均 ingest | 36.49 s | 9.95 s | -72.72% |
| 单 conversation 最大 ingest | 38.93 s | 11.30 s | -70.97% |
| construction assurance 累计 | 124.21 s | 129.52 s | +4.27% |
| 完整前台命令 | 425.80 s | 220.36 s | -48.25% |

`assurance` 包含等待/验证 L1 的时间，并发不会缩短该串行门禁。本轮其中一个 scope 的
首次 L1 返回零条，触发一次有界后台重放；最终该 scope 产生 6 条 atomic memory，未阻塞
其余 L0 写入。

### 4.2 生成式构建 token 审计

token 来自 MemoryCore 日志在各层实际发出的 input/output metric，不是字符估算。L0
embedding provider 没有暴露同口径 token，因此下表只统计 L1/L2/L3 生成式调用。

| 层 | 串行基线调用 | 基线 input / output / total | V18 调用 | V18 input / output / total |
| --- | ---: | ---: | ---: | ---: |
| L1 | 8 | 14,299 / 3,561 / 17,860 | 9 | 16,578 / 3,801 / 20,379 |
| L2 | 8 | 19,899 / 1,157 / 21,056 | 10 | 25,178 / 1,370 / 26,548 |
| L3 | 8 | 11,038 / 745 / 11,783 | 9 | 12,349 / 887 / 13,236 |
| 合计 | 24 | 45,236 / 5,463 / 50,699 | 28 | 54,105 / 6,058 / 60,163 |

本轮比基线多 9,464 token（+18.67%）。额外成本来自一次随机 L1 空输出重放，以及 dirty
scope 随后多触发的 L2/L3 聚合，不是 L0 并发机制自身需要模型调用。生产侧应继续记录
每层重放预算并对空输出率做 SLO，不能只报告成功轮次的最低 token。

最终 300 条回答均走确定性槽位编译，因此检索后送给回答模型的 prompt token、completion
token 和 total token 均为 0。524.19 是“检索得到且可审计的 evidence 字符数”，不是
实际送模 token；本轮这些 evidence 没有进入 LLM。

## 5. 本轮实现与 bad case 修复

### 5.1 Source-grounded typed episode 直达

对带可信 `query_time/timezone` 的相对时间导航回忆，先从 construction trace 的本地索引
读取类型化 episode。只有同时满足以下门禁才绕过 L0 网络查询和 ANN：

- episode 为 selected/confirmed 的活动状态；
- 构建置信度不低于 0.97；
- source lineage 是原始 source session 的子集；
- 时间区间命中，且同区间只有一个可回答候选；
- 与回答阶段同一个确定性编译器得出唯一答案。

低置信、取消、source 不匹配、缺 query time 或同时间窗多候选全部 fail closed，继续走原
检索链路。直达 hit 使用紧凑 `L0T` 证据，仍保留 source ID 供审计。

### 5.2 中文 NLU/ASR 槽位兼容

构建侧现在兼容 scalar、`{value, confidence}`、slot list 和嵌套 NLU envelope，支持
中英文 destination/address/state 槽名。结构化 top candidate 置信度门槛为 0.85，冲突
N-best 最小 margin 为 0.08；低置信结构化槽位存在时，不允许 noisy transcript 反向覆盖。

同时补充中文取消、确认、改口/supersedes、相对时间问法和 CJK+Latin 归一化，例如
`虹桥机场T2`。12 条 synthetic adapter diagnostic 全部通过：6 条安全直答，6 条按预期
拒绝，模型调用和 token 为 0。该诊断不是公开 benchmark，也不是实车麦克风/ASR 测试，
不能据此宣称真实中文座舱准确率 100%。

### 5.3 Live trace 增量刷新

首次边建边查时发现 typed 索引只缓存第一段 construction trace，后追加 conversation 不会
进入当前进程索引。修复为每次记录 construction decision 时增量更新；从磁盘重载时执行
merge 而不是覆盖。该问题不影响证据召回，但会让本应直达的请求退回 L0 时间检索。

### 5.4 引号内时间槽不再触发复杂回退

6 条天气问题的历史槽值含 `"now"`/`"right now"`。旧复杂度检测把引号中的历史内容
误判为当前查询操作符，产生无用 L2/L3 读取和 0.66--2.50 s 长尾。现在仅在路由复杂度
检测时移除 quoted anchor；未加引号的 latest/current/update 等操作仍保持回退门禁。

### 5.5 被否决的整段 L0 合批

曾尝试把多个 source session 合并为单个大 L0 transport request：

- 128 条首先被 MemoryCore 的单请求最大 100 条限制拒绝；该失败 namespace 没有成功
  写入或产生模型 token；
- 修正为 100 条后能运行，但一个 conversation 的 ingest 从串行 39.75 s 增至 86.52 s，
  增加 117.67%，原因是服务端大 embedding batch 长尾；
- 改为独立 session 并发 4 后，同一 smoke ingest 降至 10.94 s，较串行降低 72.49%。

因此代码保留大 batch 为 opt-in 实验能力，最终配置明确使用 `session` 模式和并发 4。

## 6. 剩余严格错误

| QA | 系统回答 | Gold | 审计结论 |
| --- | --- | --- | --- |
| q0047 | `The Civic Center Garage` | `Civic Center Garage` / 地址 | 仅冠词表面差异，Contains 正确 |
| q0088 | `10am` | `Next Monday.` | 问题问时间，源消息含 Monday at 10am；派生标签取错槽 |
| q0114 | `The Westin` | `The Clement Hotel` | 驾驶员明确选择 Westin；派生标签取了另一候选 |

仍按正式 Judge 报告 297/300 和 298/300，没有改数据或 Judge，也没有针对 QA ID/gold
加入特例。按源对话人工审计，后两条是派生标注冲突，但不能用人工结论替代正式分数。

## 7. 验证、产物与提交

- `framework_eval`：167/167 tests passed；
- 最终检索：`benchmarks/framework_eval_runs/cockpit-e2e-20260826-v18-live/kvret-300-l0concurrent/retrieval-final-v2.jsonl`；
- 最终预测：`benchmarks/framework_eval_runs/cockpit-e2e-20260826-v18-live/kvret-300-l0concurrent/predictions-final-v2.jsonl`；
- 最终评分：`benchmarks/framework_eval_runs/cockpit-e2e-20260826-v18-live/kvret-300-l0concurrent/score-final-v2/score-summary.json`；
- 中文诊断：`benchmarks/framework_eval_runs/cockpit-e2e-20260826-v18-live/chinese-asr-diagnostic-v1.json`。

独立代码提交：

- `16392fa perf: short-circuit grounded typed episode retrieval`
- `189662d feat: harden typed cockpit memory for Chinese ASR`
- `33e31a9 perf: batch lossless L0 cockpit transport`
- `1c97c5e fix: honor MemoryCore L0 batch limit`
- `64a20d6 perf: flush independent L0 sessions concurrently`
- `61200a5 fix: refresh typed index during live construction`
- `a2a400c perf: keep quoted temporal slots on fast path`

## 8. 下一步

下一阶段不再扩大 KVRET 分层消融，优先做两件与端侧落地直接相关的工作：

1. 在另一份公开座舱短指令 held-out 集合和私有实车流上复用同一配置，冻结阈值，不新增
   数据集品牌/QA 特例；同时分别报告 text oracle、ASR 1-best 和结构化 NLU 三条轨道。
2. 把当前 Python trace sidecar 换成进程外、小型可持久化 typed episode index，并加入
   TTL、版本、驾驶员/乘员 namespace、取消/覆盖事务和冷启动恢复；目标是保持当前
   fail-closed 门禁，同时消除进程重启后的索引重建和 Python 扫描开销。

开放问答、歧义、多实体冲突再回退端侧小模型；候选应以紧凑状态 JSON 提供，而不是把
完整 L0/L1 上下文全部交给弱模型。L2/L3 继续在车辆空闲阶段异步生成，并把随机空输出
重放率、token 和完成延迟纳入长期监控。
