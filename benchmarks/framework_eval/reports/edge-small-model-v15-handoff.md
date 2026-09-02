# TencentDB 端侧小模型记忆优化迁移手册（V15）

日期：2026-08-25

分支：`experiment/cockpit-memory-v13-20260824`

目标：在不依赖强云端模型、端侧模型能力有限的情况下，提高座舱记忆问答准确率，同时降低在线 token、模型调用次数和延迟。

## 1. 最终结论

本轮没有修改 TencentDB MemoryCore 源码或抽取提示词，没有修改 benchmark、回答系统提示词或 Judge。改动都位于评测适配层、检索编排、回答编排和运行配置。

在同一份 KVRET 300 条检索上下文上，使用本机 Ollama `llama3.2:latest`（3B 级、镜像约 2.0 GB）、并发 2、输出上限 64 token：

| 指标 | 纯 3B 回答基线 | V15：结构化直答 + 3B 回退 | 变化 |
| --- | ---: | ---: | ---: |
| Exact | 239/300（79.67%） | 258/300（86.00%） | +19 题，+6.33 pp |
| Contains | 270/300（90.00%） | 291/300（97.00%） | +21 题，+7.00 pp |
| 模型调用 | 300 | 100 | -66.67% |
| Prompt token | 140,347 | 50,442 | -64.06% |
| Completion token | 2,485 | 817 | -67.12% |
| 总 token | 142,832 | 51,259 | -64.11% |
| 回答墙钟时间 | 30.102 s | 9.665 s | -67.89% |
| 3B 路径平均 / P50 / P95 | 200.0 / 156.3 / 488.6 ms | 190.8 / 175.1 / 292.5 ms | 仅 100 条进入 |
| 确定性路径平均 / P50 / P95 | 无 | 0.397 / 0.340 / 0.595 ms | 200 条进入 |

类别结果：

| 类别 | 纯 3B Exact / Contains | V15 Exact / Contains | V15 路由 |
| --- | ---: | ---: | --- |
| Calendar（100） | 84 / 84 | 99 / 99 | 100 条结构化直答 |
| Navigation（100） | 61 / 92 | 59 / 92 | 100 条 3B 回退 |
| Weather（100） | 94 / 94 | 100 / 100 | 100 条结构化直答 |

导航 Exact 的 61→59 是两次本地生成的表面形式波动，Contains 均为 92；最终总体提升来自日历和天气槽位，不应把导航波动归因于结构化路由。

同一套 `cockpit_v1` 在 SLURP 相对时间 300 条上取得 Exact/Contains 300/300，模型调用和生成式 token 均为 0，整批回答耗时 0.101 s。该结果覆盖 18 个场景，但数据是由公开 SLURP transcript 派生的三轮文本记忆任务，不代表真实含 ASR 噪声的整车系统达到 100%。

作为参照，V14 的 DeepSeek V4 Flash 在 KVRET 300 条上为 Exact 84.33%、Contains 97.67%、总回答 token 136,773。V15 的本地 3B + 结构化路由 Exact 高 1.67 pp、Contains 低 0.67 pp，token 少 62.52%。模型和运行轨不同，因此这是工程参照，不是严格模型排行榜。

## 2. 最终在线链路

```text
用户问题 + query_time/timezone
        ↓
确定性相对时间解析（无锚点不猜）
        ↓
L0 时间候选 + 槽位重排 + Top-1 / 有界回退
        ↓
同 source session 邻域恢复，完整消息边界裁剪
        ↓
高置信槽位编译器
   ├─ 唯一且可审计：直接返回，0 LLM token
   └─ 不支持或有歧义：端侧 3B 回答
        ↓
仍不确定时才允许云端模型升级（当前评测未启用）
```

核心思想不是“让小模型学会更复杂推理”，而是把记忆系统能确定完成的工作前移：身份/时间/会话隔离、证据选择、状态恢复、槽位解析和输出约束由确定性组件承担；小模型只处理真正开放或歧义的问题。

## 3. 一路上的有效优化

### 3.1 先修评测与数据契约，再调检索

早期低分混入了评测协议错误：session 时间没有送到答案阶段、speaker 和 source turn ID 在 TencentDB v3 写入时丢失、Full Context 被静默截断、回答器没有加载正式配置。先统一了以下契约：

- 每条 L0 保留 `[source_turn_id] [source_time=...] Speaker: content`。
- `MemoryHit.metadata` 始终携带 timestamp/authored_at/session_timestamp。
- 回答上下文重新渲染来源时间，避免模型看到“昨天”却不知道对话日期。
- namespace 清单显式绑定 `team_id/user_id/agent_id/task_id`；每个评测 conversation 独立隔离。
- 多人物长对话按人物 perspective 隔离；单驾驶员座舱数据使用 `single`，但 speaker 仍逐条保真。

迁移到私有数据时，先检查这五项。否则后面的 Top-K、Prompt 或模型比较会把协议错误误判为记忆框架能力。

### 3.2 L0 同会话邻域解决“命中助手问题、漏掉用户回答”

座舱短指令常见结构是：

```text
Driver: 设置闹钟
Assistant: 几点？
Driver: 下午五点
```

向量检索更容易命中语义丰富的助手追问，而答案在后一条极短的用户回复中。最终策略是：

- backend 先取更宽候选池，但不直接扩大送模上下文；
- 命中 L0 后只在同一 backend/source session 内展开；
- `before=2`、`after=12`；
- source session 不超过 8 条时保留完整 episode，长会话仍围绕锚点裁剪；
- 不允许跨 session 拼邻居。

KVRET evidence 从 409/420→419/420→420/420；SLURP 三轮指令也由该策略稳定覆盖。关键不是盲目增加 Top-K，而是恢复命中消息的交互状态。

### 3.3 选择性构建，L0 保真、L1 少而准、L2/L3 异步

最终座舱构建配置遵循：

- 所有源消息无损进入 L0；
- 只有 durable、pending、profile、update、expiry 类型 episode 进入 L1；
- L1 按 conversation 合批，而不是每个零碎 session 都调用一次模型；
- 语义空 L1 最多补偿一次，之后回到 `l0_only`，不阻塞在线路径；
- L2/L3 只处理 dirty scope，并在车辆空闲/停车时异步聚合；
- 当前 L2/L3 共享 agent 锁，单 endpoint 内 worker 保持 1，并行放在不同 namespace/shard 之间。

KVRET V13 的 420 个 session 中只有 38 个进入 L1（9.05%），触发字符为 8.49%，最终 L1/L2/L3 构建共 74,853 token。早期对 138 个碎片 session 无选择地运行四层构建时，L1/L2/L3 共 376,888 token，其中 L2/L3 比 L1 又增加 187,064 token，却没有提高该槽位任务的 evidence recall。

不要把这个数字直接当成完全同数据集的严格消融，但方向很稳定：短期槽值查 L0；偏好、习惯、跨 session 关系才值得 L1/L2/L3。

### 3.4 Top-1 快路由与有界回退

最终检索不是固定 L0/L1/L2/L3 全塞：

- 宽候选池做槽位、日期、说话人和 episode 完整性重排；
- 高置信单事实走 L0 Top-1；
- 缺关键槽、更新问法、多事实问法才回退 Top-3 + L1；
- 画像/关系问题再按需取 L2/L3；
- fast/fallback 上下文分别限制为 900/2,200 字符。

V13 KVRET 中 413/420 走 Top-1，只有 7 条回退；平均上下文 706.8 字符。V14 相对时间 300 条中，时间 short circuit 后 294 条无需动态 ANN 搜索。

### 3.5 相对时间必须以请求元信息为锚

请求建议至少携带：

```json
{
  "query_time": "2026-08-25T09:30:00+08:00",
  "timezone": "Asia/Shanghai"
}
```

处理规则：

- 支持昨天、昨天上午、前天、前天晚上、last Tuesday 等半开时间区间；
- 优先使用可信 `query_time/question_time/request_time/asked_at/captured_at`；
- 无时间锚点时不猜绝对日期；
- 同时匹配 source mention time 与语义 event time；
- 在 ANN Top-K 之前加入时间候选；
- 高置信时间候选直接 short circuit；
- 只向有相对时间且可解析的问题添加紧凑 user envelope，系统提示不变。

V14 相对显式日期基线，KVRET/SLURP 平均检索延迟分别下降 82.76%/81.56%，动态搜索调用下降 96.08%/89.87%。

### 3.6 完整消息边界比机械字符截断更重要

真实 bad case `q0268` 的 900 字符硬截断把：

```text
Setting navigation to Webster Garage.
```

截成 `Setting navigation to Webster G`。修复策略不是全局加大上下文，而是：

- 若预算落在一条消息中间，最多允许 160 字符补齐当前行；
- 当前行剩余超过 160 字符时退回上一完整行；
- 记录 `context_budget_overflow_chars` 和 `context_boundary_action`。

在 KVRET 300 条上，78 条补齐行尾，溢出平均 44.53、最大 118 字符；摊到全部问题，平均上下文仅从 718.31 增至 729.89 字符，即 +11.58（+1.61%）。`Webster Garage` 只需多 7 个字符即可恢复。

### 3.7 高置信槽位编译器让弱模型少做它不擅长的事

小模型 bad case 表明：即使 evidence 100% 正确进入上下文，3B 仍可能输出证据头、时间注释，或对清晰地点回答 “Insufficient evidence”。V15 在答案模型之前加入三种严格形态：

1. `command → assistant clarification → user reply`；
2. 指定 speaker、事件 anchor 和目标日期的 calendar time；
3. 指定 driver、forecast anchor 和目标日期的 weather location。

每个候选都必须满足：

- 问题意图匹配；
- 可信日期可解析；
- 同一 source session；
- speaker 匹配；
- quoted anchor/slot 匹配；
- 候选唯一，或有唯一 anchor/assistant echo 支持；
- 输出携带 source message ID、reason、confidence；
- 不读取 gold answer 或 benchmark evidence ID。

任何歧义都返回 `None` 并调用小模型。KVRET 中日历 100 条、天气 100 条进入直答，导航 100 条全部回退；SLURP 300 条全部进入 clarification 直答。

## 4. 典型 bad case 与修复方法

| Bad case | 根因 | 修复 | 门禁 |
| --- | --- | --- | --- |
| 检索命中 “What time?”，漏掉 “five pm” | 助手提问语义强、用户回复太短 | 同 source session 前后展开，短 episode 完整保留 | session provenance 必须存在 |
| “昨天上午/前天晚上”低召回 | 查询时间未归一化，ANN 不理解相对日期 | `query_time + timezone`→绝对区间，Top-K 前时间候选 | 无锚点不解析 |
| 引号中的 `"today"` 被当成本次查询时间 | 历史槽位内容与时间操作符混淆 | 引号内相对词不参与 query-time 解析 | 仍保留原文供回答 |
| `Webster Garage`→`Webster G` | 900 字符在消息中间硬切 | 160 字符内补齐当前行 | 超限则退回上一行 |
| 3B 对 `Will there be frost in Durham next week?` 回答证据不足 | 小模型抽取能力波动 | 天气地点状态编译并验证 speaker/date/echo | 多候选同分回退模型 |
| 3B 输出 `[resolved_query_time: ...]` 而非 `3 pm` | 小模型复制了控制信息 | 日历时间槽按 speaker + event anchor 唯一解析 | 多个 anchor 时间则回退 |
| `home` 被回答为家庭地址 | 语义正确但 strict gold 只接受别名 | 报告 strict 与 semantic 分离；后续做 POI alias state | 不改 Judge 掩盖问题 |
| Westin/Clement、Coffee/Cafe | 源对话与 gold 冲突或拼写不一致 | 标记数据问题，不用规则强改成 gold | 保留源证据审计 |
| 问题问 time、gold 却是 `Next Monday` | benchmark 标签语义不一致 | 系统仍返回证据中的 `10am` | 单独报告，不为分数过拟合 |
| 缺 construction trace 时跨 session 扩窗 | conversation batch 的 transport session 不等于 source session | provenance 不完整时 fail closed，不展开 | 禁止猜 source session |
| 空 L1 无限补偿 | “语义上无需长期记忆”被误判为失败 | 区分 clean/semantic-empty/failed，最多一次补偿 | 在线先返回 L0 |

## 5. 失败或被回滚的尝试

这些经验值得在另一台电脑直接避开：

1. **只提高 Top-K/上下文预算。** LoCoMo Top-K 18 让上下文增加约 6%，recall 只增 0.365 pp，新增 evidence 对应题原本已答对。更大的 context 还会拖累弱模型注意力。
2. **全局 source-date 重排。** 54 个受影响问题中 2 个增益、3 个回退，最终回滚。日期应是门控/候选特征，不应无条件压过语义与事件状态。
3. **固定 radius 邻域。** 锚点落在第 4 条时仍可能漏掉第 1 条发起指令；短 session 应完整保留，长 session 才使用半径。
4. **所有短指令都构建 L1/L2/L3。** 约一半碎片 session 的 L1 调用为空，L2/L3 在槽位任务无 recall 收益，却接近翻倍构建 token。
5. **同步等待 L2/L3 或空 L1 修复。** 共享 agent lock 会产生 profile lock conflict，在线延迟不可控。修复必须后台化。
6. **只看 evidence recall。** 多数最终 bad case 已 100% 召回，错误发生在答案抽取、别名或数据冲突；继续调 retrieval 不会解决。
7. **只看 Exact/F1。** POI + 地址可能语义正确但 Exact 低；Judge、Contains、Exact、evidence 和人工 bad-case 分类要一起看。
8. **把 replay 说成 live 重跑。** 当前 V15 完整消息实验因本机 MemoryCore 要求 Bearer key、但容器 key 为空，使用冻结 hits 重渲染上下文；报告中明确标为 paired replay。不要隐藏这一运行边界。
9. **并发不带断点续跑。** 本地/远端模型都可能中途断开；输出必须按 QA ID 去重，使用 `--resume`，不要重算已完成问题。

## 6. 另一台电脑的迁移执行顺序

### 6.1 先备份和建分支

```bash
git status --short
git switch -c experiment/edge-cockpit-memory
git tag before-edge-cockpit-v15
```

私有数据、密钥、namespace manifest 和运行产物不要提交。每个独立优化一个 commit，先小样本门禁再扩大。

### 6.2 按顺序迁移适配层提交

最稳妥的方式是获取整个 `experiment/cockpit-memory-v13-20260824` 分支；如果两台电脑共享基线 `11ffa75`，可以迁移该基线之后的连续提交。不要只挑最后五个 V15 提交，因为它们依赖此前的 session provenance、选择性构建和自适应检索。

选择性构建与座舱检索基础，按时间顺序为：

```text
2e8e0a8 add adaptive L0-first retrieval
47654c2 add selective L1 construction policy
15ce1f1 buffer dirty L2 L3 construction
d146e4f defer unused adaptive memory layers
d737f69 make async profiles nonblocking
880ee49 batch fragmented cockpit memory builds
ffb90fc bound L1 extraction microbatches
bba5fd9 defer profiles until L1 microbatches drain
f9c8c1d compact fragmented cockpit L1 input
db583c4 enable L1 compaction during ingest
5246e69 route explicit dates to source sessions
bdf5f24 recover empty selective L1 construction
b874bd9 distinguish semantic empty L1 scopes
6798ba3 move empty L1 recovery off request path
7f4054b add scene-aware episode write policy
e8dfeff add slot-aware adaptive retrieval
3dc782c add final cockpit endpoint profile
1277d41 retain initiating cockpit slot turn
9c5c0c3 preserve complete short cockpit episodes
```

时间与检索：

```text
15a1b8b normalize anchored temporal expressions
8d512b9 retrieve temporal candidates before top-k
41c01bf inject temporal anchors at answer time
d04a45b fast-route bounded temporal queries
9a9a154 short-circuit confident temporal retrieval
2beb3ff ignore quoted relative-time slot values
b7329f5 guard L0 windows without session provenance
```

弱模型回答：

```text
ebb24ae extract grounded cockpit slot replies
469d66a bypass answer model for grounded slots
4b256ba preserve complete evidence lines within budget
22b6ded compile grounded cockpit slots before model fallback
bc1f879 report answer latency by edge route
```

如果另一台电脑的代码已经分叉，不要强行整段 cherry-pick；按文件迁移并逐个测试：

- `benchmarks/framework_eval/temporal.py`
- `benchmarks/framework_eval/adapters/adaptive_policy.py`
- `benchmarks/framework_eval/adapters/tencentdb_http.py`
- `benchmarks/framework_eval/runner.py`
- `benchmarks/framework_eval/answering.py`
- `benchmarks/framework_eval/cockpit_slots.py`
- `benchmarks/framework_eval/configs/tencentdb_cockpit_e2e_v2.env`

### 6.3 生产数据最小契约

每条消息至少保存：

```json
{
  "message_id": "vehicle-session-42-turn-03",
  "session_id": "vehicle-session-42",
  "user_id": "stable-private-user-id",
  "role": "user",
  "content": "明天上午导航去虹桥机场",
  "mentioned_at": "2026-08-25T09:30:00+08:00",
  "event_time": {
    "start": "2026-08-26T06:00:00+08:00",
    "end": "2026-08-26T12:00:00+08:00"
  },
  "timezone": "Asia/Shanghai"
}
```

`mentioned_at` 和 `event_time` 必须区分；“昨晚说下周一去机场”有两个不同时间。session 是一次交互，不是用户身份。L0/L1/L2/L3 应共享同一稳定 user/agent subject，避免画像跨用户污染。

### 6.4 启用最终配置

```bash
source benchmarks/framework_eval/configs/tencentdb_cockpit_e2e_v2.env
export TDAI_EVAL_ISOLATION_MAP=/abs/path/to/isolation.json
export TDAI_EVAL_CONSTRUCTION_TRACE=/abs/path/to/construction.jsonl
export TDAI_EVAL_USER_KEY_FILE=/abs/path/to/user-key
```

关键值是：选择性 L1、conversation batch、异步 dirty L2/L3、adaptive v2、Top-1、900/2,200 字符、160 字符完整行溢出、L0 before=2/after=12/max=8、temporal interval + short circuit、`cockpit_v1` 回答路由。

### 6.5 冻结 300–500 条 held-out 清单

不要用全量私有数据边看结果边调。建议：

- 开发集 100–200 条用于 bad case；
- held-out 300–500 条只在阶段完成后运行；
- 按导航、天气、日程、媒体、电话、车控、偏好、跨 session 更新分层抽样；
- 保留 ASR 原文、规范文本和时间元信息；
- selection manifest 固定 question ID，所有版本保持同一顺序。

### 6.6 跑检索、校验、回答和评分

```bash
python3 -m benchmarks.framework_eval.cli retrieval \
  --adapter tencentdb \
  --dataset YOUR_DATASET_PLUGIN \
  --dataset-root /abs/path/to/private-normalized-data \
  --selection-manifest /abs/path/to/frozen-selection.json \
  --top-k 8 \
  --output /abs/path/to/run/retrieval.jsonl \
  --base-url http://127.0.0.1:8420 \
  --skip-ingest

python3 -m benchmarks.framework_eval.cli validate \
  --retrieval /abs/path/to/run/retrieval.jsonl \
  --expected-count 300 \
  --output /abs/path/to/run/validation.json

export MEMEVAL_ANSWER_BASE_URL=http://127.0.0.1:11434/v1
export MEMEVAL_ANSWER_API_KEY=ollama
export MEMEVAL_ANSWER_MODEL=llama3.2:latest
export MEMEVAL_ANSWER_MAX_TOKENS=64
export MEMEVAL_ANSWER_TEMPORAL_QUERY_MODE=interval_v1
export MEMEVAL_ANSWER_DETERMINISTIC_SLOT_MODE=cockpit_v1

python3 -m benchmarks.framework_eval.cli answer \
  --input /abs/path/to/run/retrieval.jsonl \
  --output /abs/path/to/run/predictions.jsonl \
  --concurrency 2 \
  --resume

python3 -m benchmarks.framework_eval.cli score \
  --dataset YOUR_DATASET_PLUGIN \
  --dataset-root /abs/path/to/private-normalized-data \
  --input /abs/path/to/run/predictions.jsonl \
  --output-dir /abs/path/to/run/score \
  --metrics exact contains \
  --resume
```

先用并发 1 做 1–10 条烟测，再尝试 2。端侧吞吐由内存带宽、上下文长度和模型 runtime 决定，不要直接复制云 API 的并发 8。

代码回归命令：

```bash
python3 -m unittest discover \
  -s benchmarks/framework_eval \
  -t . \
  -p 'test_*.py' \
  -v
```

### 6.7 每轮必须记录的指标

- 构建：L1/L2/L3 调用数、prompt/completion token、空输出、重试、dead letter、ready 时间；
- 检索：evidence recall、实际截断后 evidence、Top-1/回退比例、动态搜索调用、P50/P95；
- 上下文：mean/P95/max 字符与 token、截断数、完整行溢出数；
- 回答：模型调用数、确定性直答数、每条 route reason/source IDs、prompt/completion token、P50/P95；
- 质量：Exact、Contains、语义 Judge、类别分数、正确→错误回退数；
- 费用：构建、回答、Judge、embedding 分开。字符只能估算，正式结果优先使用 provider usage。

口径必须固定：`平均上下文字符 = sum(len(最终送答题器的 rendered memory context)) / 问题数`，只包含经过重排、邻域恢复和预算裁剪后的记忆文本及其来源标记，不包含 system prompt、原始问题、时间 envelope 和模型输出。`prompt_tokens` 才是 provider 对整个输入请求返回的 token 用量；构建 token、回答 token、Judge token 和 embedding 成本不得混加后再声称是“检索 token”。检索耗时、回答 route 耗时和整批墙钟时间也要分别报告。

## 7. 私有数据 bad-case 优化流程

另一台电脑不能共享数据没有关系，应在本地把每个问题按 QA ID 联结四份产物：question、retrieval、prediction、score，然后按顺序分类：

1. **无 evidence：** 查 namespace、时间过滤、候选召回和 speaker/session provenance；
2. **hits 有、context 无：** 查 Top-K 竞争和字符/消息边界截断；
3. **context 有、答案错：** 查槽位抽取、别名、更新状态和小模型输出；
4. **答案语义对、Judge 错：** 查 gold 冲突、格式和 alias，但不要直接改 Judge；
5. **正确题回退：** 新规则必须回滚或提高置信门槛。

每个规则需要保存：触发意图、使用字段、候选、source IDs、置信度、拒绝原因。开发集上的新增正确数不能抵消 held-out 的正确→错误回退；高置信直答优先追求 precision≥99%，coverage 是第二目标。

## 8. 下一步最值得做的优化

### P0：把查询时正则升级为构建时 typed cockpit episode

当前 `cockpit_slots.py` 已证明“结构化槽位 + 可审计来源”有效，但它仍从文本上下文解析。生产版建议在写入 L0 时同步生成低成本结构：

```json
{
  "intent": "navigation.set_destination",
  "slots": {"poi": "虹桥机场"},
  "state": "confirmed",
  "mentioned_at": "...",
  "event_time": {"start": "...", "end": "..."},
  "source_ids": ["T01", "T02"],
  "confidence": 0.98,
  "supersedes": null
}
```

优先复用车机 NLU/ASR 已有 intent 和 slot，不要再用大模型重抽一遍；只有低置信或开放文本才交给小模型。这样规则从英文问句模板中解耦，也更适合中文真实数据。

### P0：导航 POI 状态机

当前剩余 9 个 Contains bad case 大多在导航。下一步应记录：候选 POI→用户选择→助手确认→地址/别名映射→路线更新。答案可以按问题要求返回 canonical POI、address 或用户别名 `home/work`。必须解决：

- `home` 与家庭地址的双向 alias；
- `Ravenswood` 与 `Ravenswood Shopping Center` 的实体链接；
- “先推荐 A，用户改选 B”的最终状态；
- POI 名拼写变化 Coffee/Cafe；
- assistant 只返回地址时，从已确认 POI 关联地址。

先在开发集构建状态机，再在未见过的导航对话上门禁，避免为 KVRET 名称写白名单。

### P1：中文、ASR 和端侧 NLU 鲁棒性

- 数字、日期、上午/下午、二十四小时制统一；
- ASR 同音词、地名别名、口语停顿和纠错；
- 多轮 `不对/改成/还是去` 形成 supersedes 链；
- 置信度低时保留原 ASR 与 N-best，不要覆盖 L0；
- 用 0.5B/1.5B 模型做 constrained JSON slot extraction，失败再回退 3B。

### P1：分级模型升级

建议线上路由：

```text
确定性 typed slot
  → 0.5B/1.5B 约束抽取
  → 3B 证据问答
  → 云端模型（仅复杂关系/歧义且用户允许）
```

每级都带置信度和超时；失败向上升级，不要让低置信规则静默输出。

### P2：只为长期问题构建高层记忆

L2/L3 保留给偏好、习惯、跨人物关系、事件去重、长期计划和因果总结。LoCoMo v7/v8 已证明 source-grounded relation/event rollup 能提高复杂题，但这类构建是一次性额外成本，必须异步、可审计且按需检索，不能重新污染短指令路径。

## 9. 汇报边界

- KVRET/SLURP 是公开 transcript 派生的文本记忆任务，不包含真实麦克风、ASR 和端侧资源抖动；
- 当前时间日期是为稳定排序构造的，不是原数据录音日期；
- V15 完整行上下文使用冻结 hits 的 paired replay，因为本机 live MemoryCore 鉴权配置为空；
- 生成式 token 不含 embedding；
- “TencentDB 原生能力”和“MemoryCore + adapter-side temporal/slot compiler”应分开命名；
- 不应将单个 conv 上通过 bad-case 调优得到的 LoCoMo 分数当作跨用户泛化结论。

## 10. 关键代码和产物

- 最终配置：`benchmarks/framework_eval/configs/tencentdb_cockpit_e2e_v2.env`
- 槽位编译器：`benchmarks/framework_eval/cockpit_slots.py`
- 回答路由与逐 route 延迟：`benchmarks/framework_eval/answering.py`
- 完整消息预算：`benchmarks/framework_eval/runner.py`
- 时间解析：`benchmarks/framework_eval/temporal.py`
- TencentDB 自适应检索：`benchmarks/framework_eval/adapters/tencentdb_http.py`
- V14 时间实验：`benchmarks/framework_eval/reports/temporal-cockpit-v14-end-to-end.md`
- V13 构建实验：`benchmarks/framework_eval/reports/kvret-v13-deepseek-flash-end-to-end.md`

V15 机器可读结果位于未提交的运行目录：

```text
benchmarks/framework_eval_runs/cockpit-e2e-20260825-v15-edge/
  kvret-300/
    retrieval-complete-lines-v1.jsonl
    predictions-llama32-3b-baseline-v2-latency.jsonl
    answer-summary-llama32-3b-baseline-v2-latency.json
    score-llama32-3b-baseline-v2-latency/
    predictions-llama32-3b-cockpit-v2-latency.jsonl
    answer-summary-llama32-3b-cockpit-v2-latency.json
    score-llama32-3b-cockpit-v2-latency/
  slurp-300/
    predictions-cockpit-v1.jsonl
    score-cockpit-v1/
```

最终单元测试：147/147 通过。
